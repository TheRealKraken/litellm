mod fetch;
mod live;
mod rows;

use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use litellm_host_python::to_py;
use litellm_model_catalog::{
    CostMap, IntegrityLimits, Pass, Placed, Provenance, Refused, RegistrationTable, Registrations,
    Store, write_pass,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) use fetch::{NativeCostMapSource, NativeFetchFailed, NativeFetchPolicy, NativeFetched};
use live::LiveDict;

/// The installed document and the `register_model` calls kept above it, each as the dict Python merged
struct Inputs {
    cost_map: CostMap,
    registrations: Registrations<Py<PyDict>>,
}

impl Inputs {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            cost_map: self.cost_map.clone(),
            registrations: self.registrations.map(|kept| kept.clone_ref(py)),
        }
    }
}

/// The cost map's inputs with the name index over the dict Python serves as `litellm.model_cost`. Python owns that
/// dict and passes it to every call that reads or writes it, so this never keeps a second copy of the live rows. No
/// lock is held while Python code runs, since that code may release the GIL to a thread that then waits on the same
/// lock
#[pyclass(
    frozen,
    name = "NativeModelCostMap",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeModelCostMap {
    inputs: RwLock<Inputs>,
    limits: IntegrityLimits,
    live: RwLock<Option<Arc<Store<LiveDict>>>>,
}

#[pyclass(
    frozen,
    name = "NativeInstalled",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeInstalled {
    /// The new live table, the document without registrations. `reapply` writes them once Python publishes it, since
    /// where each one lands is a lookup in the published table
    #[pyo3(get)]
    rows: Py<PyDict>,
    /// How many keys the document itself serves, aliases included and registrations left out
    #[pyo3(get)]
    model_count: usize,
    /// The document's `fallback_generalizations` block as written, `None` when it has none
    #[pyo3(get)]
    fallback_generalizations: Py<PyAny>,
}

#[pyclass(
    frozen,
    name = "NativeRejected",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeRejected {
    #[pyo3(get)]
    reason: String,
    /// The document parsed but has fewer models than the integrity limits allow
    #[pyo3(get)]
    integrity: bool,
}

/// A finished registration pass, with the registrations no built-in row matched, by registered key
#[pyclass(
    frozen,
    name = "NativeRegistered",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeRegistered {
    #[pyo3(get)]
    unmatched: Py<PyDict>,
}

/// A registration pass that stopped because the table it was handed is no longer the host's published one
#[pyclass(
    frozen,
    name = "NativeTableSwapped",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeTableSwapped;

/// The inputs at one moment, for a caller that has to put them back, such as test isolation
#[pyclass(
    frozen,
    name = "NativeModelCostMapState",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeModelCostMapState(Inputs);

#[pyclass(
    frozen,
    name = "NativeProvenance",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeProvenance {
    #[pyo3(get)]
    revision: String,
    #[pyo3(get)]
    etag: Option<String>,
}

#[pymethods]
impl NativeModelCostMap {
    /// Starts from the bundled document. A fetched document with fewer than `min_model_count` models, or fewer than
    /// `max_shrink_ratio` times the bundled count, is rejected
    #[staticmethod]
    fn bundled(min_model_count: i64, max_shrink_ratio: f64) -> PyResult<Self> {
        let min_model_count = usize::try_from(min_model_count).map_err(|_| {
            PyValueError::new_err(format!(
                "min_model_count must be 0 or more, got {min_model_count}"
            ))
        })?;
        let limits = IntegrityLimits::new(min_model_count, max_shrink_ratio).ok_or_else(|| {
            PyValueError::new_err(format!(
                "max_shrink_ratio must be between 0 and 1, got {max_shrink_ratio}"
            ))
        })?;
        Ok(Self {
            inputs: RwLock::new(Inputs {
                cost_map: CostMap::bundled(),
                registrations: Registrations::default(),
            }),
            limits,
            live: RwLock::new(None),
        })
    }

    #[pyo3(signature = (source, policy))]
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        source: PyRef<'_, NativeCostMapSource>,
        policy: PyRef<'_, NativeFetchPolicy>,
    ) -> PyResult<Bound<'py, PyAny>> {
        fetch::fetch(py, &source, &policy)
    }

    #[pyo3(signature = (source, policy, resume_from = None))]
    fn fetch_blocking(
        &self,
        py: Python<'_>,
        source: PyRef<'_, NativeCostMapSource>,
        policy: PyRef<'_, NativeFetchPolicy>,
        resume_from: Option<PyRef<'_, NativeFetchFailed>>,
    ) -> PyResult<Py<PyAny>> {
        fetch::fetch_blocking(py, &source, &policy, resume_from.as_deref())
    }

    /// Accepts a fetched document and returns the live table built from it, or why it was rejected, in which case
    /// nothing changed
    #[pyo3(signature = (body, etag = None))]
    fn install(&self, py: Python<'_>, body: &[u8], etag: Option<String>) -> PyResult<Py<PyAny>> {
        let read = crate::logger::capture(py).scope(|| CostMap::read(body, etag, self.limits));
        match read {
            Ok(cost_map) => self.install_document(py, cost_map),
            Err(refused) => Py::new(
                py,
                NativeRejected {
                    reason: litellm_error::Report::new(&refused).to_string(),
                    integrity: matches!(refused, Refused::Integrity(_)),
                },
            )
            .map(Py::into_any),
        }
    }

    /// Installs the bundled document, which is never size checked
    fn install_bundled(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.install_document(py, CostMap::bundled())
    }

    /// Writes each entry into `rows` as main's `register_model` does. With `persist`, the entries are also kept and
    /// written again over every later install, unless writing them raised. Reports a swapped table when `rows` stopped
    /// being the host's published table partway through, so the caller writes them again into the new one
    #[pyo3(signature = (rows, entries, persist, custom_llm_provider = None))]
    fn register(
        &self,
        rows: &Bound<'_, PyDict>,
        entries: &Bound<'_, PyDict>,
        persist: bool,
        custom_llm_provider: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let py = rows.py();
        let entries: Vec<(String, Bound<'_, PyAny>)> = entries
            .iter()
            .map(|(key, value)| Ok((key.str()?.to_string(), value)))
            .collect::<PyResult<_>>()?;
        let kept = persist
            .then(|| {
                entries
                    .iter()
                    .map(|(key, registration)| {
                        let previous = self.kept(py, key);
                        Ok((key.as_str(), rows::kept(previous.as_ref(), registration)?))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let pass = self.write_pass(rows, &entries, custom_llm_provider)?;
        if let Some(kept) = kept {
            let mut inputs = self.write_inputs();
            for (key, kept) in kept {
                inputs.registrations.register(key, kept.unbind());
            }
            inputs.registrations.record(pass.landed());
        }
        into_py(py, pass)
    }

    /// Writes every kept registration into `rows` again, oldest first, so it wins the fields it shares with rows the
    /// router replayed after an install
    fn reapply(&self, rows: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let py = rows.py();
        let entries: Vec<(String, Bound<'_, PyAny>)> = self
            .read_inputs()
            .registrations
            .entries()
            .map(|(key, kept)| (key.to_owned(), kept.bind(py).clone().into_any()))
            .collect();
        let pass = self.write_pass(rows, &entries, None)?;
        self.write_inputs().registrations.record(pass.landed());
        into_py(py, pass)
    }

    /// The installed row under `key` with the kept registrations that apply to it, without anything the router wrote
    fn registered_row<'py>(
        &self,
        py: Python<'py>,
        key: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let (cost_map, applied) = {
            let inputs = self.read_inputs();
            let applied: Vec<Bound<'py, PyDict>> = inputs
                .registrations
                .over(&[key])
                .into_iter()
                .map(|kept| kept.bind(py).clone())
                .collect();
            (inputs.cost_map.clone(), applied)
        };
        rows::registered_row(py, &cost_map, key, &applied)
    }

    fn registered_keys(&self) -> Vec<String> {
        let inputs = self.read_inputs();
        rows::registered_keys(
            &inputs.cost_map,
            inputs.registrations.entries().map(|(key, _)| key),
        )
    }

    /// The installed document as a table, without any registration
    fn loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cost_map = self.read_inputs().cost_map.clone();
        rows::table(py, &cost_map)
    }

    /// The key itself when `rows` has it, otherwise the key that differs from `name` only in case
    fn find_key(&self, rows: &Bound<'_, PyDict>, name: &str) -> Option<String> {
        self.with_store(rows, |store| store.find_key(name))
    }

    /// Keys whose row's `litellm_provider` is exactly `provider`, in row order
    fn keys_in(&self, rows: &Bound<'_, PyDict>, provider: &str) -> Vec<String> {
        self.with_store(rows, |store| store.keys_in(provider))
    }

    /// Drops the name index, for a caller that changed keys in a way the dict's length does not show
    fn invalidate(&self) {
        if let Some(store) = self
            .live
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            store.invalidate();
        }
    }

    fn snapshot(&self, py: Python<'_>) -> NativeModelCostMapState {
        NativeModelCostMapState(self.read_inputs().clone_ref(py))
    }

    /// Puts back the document, registrations and provenance a `snapshot` saw
    fn restore(&self, py: Python<'_>, state: PyRef<'_, NativeModelCostMapState>) {
        *self.write_inputs() = state.0.clone_ref(py);
        self.invalidate();
    }

    #[getter]
    fn provenance(&self) -> NativeProvenance {
        let Provenance { revision, etag } = self.read_inputs().cost_map.provenance().clone();
        NativeProvenance { revision, etag }
    }
}

impl NativeModelCostMap {
    fn read_inputs(&self) -> RwLockReadGuard<'_, Inputs> {
        self.inputs.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_inputs(&self) -> RwLockWriteGuard<'_, Inputs> {
        self.inputs.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn kept<'py>(&self, py: Python<'py>, key: &str) -> Option<Bound<'py, PyDict>> {
        self.read_inputs()
            .registrations
            .get(key)
            .map(|kept| kept.bind(py).clone())
    }

    fn install_document(&self, py: Python<'_>, cost_map: CostMap) -> PyResult<Py<PyAny>> {
        self.write_inputs().cost_map = cost_map.clone();
        let rows = rows::table(py, &cost_map)?;
        let fallback_generalizations = match cost_map.fallback_generalizations() {
            Some(block) => to_py(py, block)?,
            None => py.None(),
        };
        *self.live.write().unwrap_or_else(PoisonError::into_inner) =
            Some(Arc::new(Store::new(LiveDict::new(&rows))));
        Py::new(
            py,
            NativeInstalled {
                model_count: rows.len(),
                rows: rows.unbind(),
                fallback_generalizations,
            },
        )
        .map(Py::into_any)
    }

    fn write_pass<'a, 'py>(
        &self,
        rows: &Bound<'py, PyDict>,
        entries: &'a [(String, Bound<'py, PyAny>)],
        custom_llm_provider: Option<&str>,
    ) -> PyResult<Pass<'a, Bound<'py, PyAny>>> {
        write_pass(
            &mut HostTable {
                map: self,
                rows,
                custom_llm_provider,
            },
            entries,
        )
    }

    /// Queries the index over `rows`, rebinding it when `rows` is not the dict it was built over
    fn with_store<T>(
        &self,
        rows: &Bound<'_, PyDict>,
        query: impl FnOnce(&Store<LiveDict>) -> T,
    ) -> T {
        let bound = self
            .live
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .filter(|store| store.rows().is(rows));
        let store = bound.unwrap_or_else(|| {
            let store = Arc::new(Store::new(LiveDict::new(rows)));
            *self.live.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&store));
            store
        });
        query(&store)
    }
}

/// The dict Python publishes as `litellm.model_cost`, written through the host's own lookup and merge. No lock is
/// held while the host runs, since its lookup reaches `find_key` on this map
struct HostTable<'a, 'py> {
    map: &'a NativeModelCostMap,
    rows: &'a Bound<'py, PyDict>,
    custom_llm_provider: Option<&'a str>,
}

impl<'py> RegistrationTable for HostTable<'_, 'py> {
    type Registration = Bound<'py, PyAny>;
    type Landing = rows::Landing<'py>;
    type Error = PyErr;

    fn resolve(
        &mut self,
        key: &str,
        registration: &Self::Registration,
    ) -> PyResult<Option<Self::Landing>> {
        rows::resolve(self.rows, key, registration, self.custom_llm_provider)
    }

    fn write(
        &mut self,
        landing: Self::Landing,
        registration: &Self::Registration,
    ) -> PyResult<Placed> {
        rows::write(self.rows, &landing, registration)?;
        self.map.invalidate();
        rows::changed(self.rows.py())?;
        Ok(Placed {
            key: landing.key,
            matched_builtin: landing.matched_builtin,
        })
    }
}

fn into_py<'py>(py: Python<'py>, pass: Pass<'_, Bound<'py, PyAny>>) -> PyResult<Py<PyAny>> {
    match pass {
        Pass::Swapped { .. } => Py::new(py, NativeTableSwapped).map(Py::into_any),
        Pass::Finished { unmatched, .. } => {
            let table = PyDict::new(py);
            for (key, registration) in unmatched {
                table.set_item(key, registration)?;
            }
            Py::new(
                py,
                NativeRegistered {
                    unmatched: table.unbind(),
                },
            )
            .map(Py::into_any)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use pyo3::ffi::c_str;
    use rstest::rstest;

    use super::*;

    const HOST: &std::ffi::CStr = c_str!(
        r#"
from types import SimpleNamespace

class Row(dict):
    names = ()

found = []

def resolve(rows, key, registration, custom_llm_provider):
    found.append(cost_map.find_key(rows, key))
    return SimpleNamespace(key=key, base={}, matched_builtin=True)

def merged(current, base, registration):
    return Row({**(current or {}), **base, **registration})

def changed():
    pass
"#
    );

    #[rstest]
    fn a_host_lookup_that_reaches_find_key_on_the_same_map_finishes() {
        Python::initialize();
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let outcome = Python::attach(|py| -> PyResult<(Vec<Option<String>>, f64)> {
                let cost_map = Py::new(py, NativeModelCostMap::bundled(0, 0.0)?)?;
                let module_name = std::ffi::CString::new(rows::MODULE).unwrap();
                let host =
                    PyModule::from_code(py, HOST, c_str!("host_model_cost.py"), &module_name)?;
                host.setattr("cost_map", &cost_map)?;
                let modules = py.import("sys")?.getattr("modules")?;
                let packages = ["litellm", "litellm.rust_bridge", "litellm.rust_bridge.host"];
                for package in packages {
                    modules.set_item(package, PyModule::new(py, package)?)?;
                }
                modules.set_item(rows::MODULE, &host)?;
                let rows = PyDict::new(py);
                rows.set_item("Lit-Reentry", PyDict::new(py))?;
                let entry = PyDict::new(py);
                entry.set_item("input_cost_per_token", 1.5)?;
                let entries = PyDict::new(py);
                entries.set_item("lit-reentry", entry)?;
                let registered = cost_map
                    .bind(py)
                    .call_method1("register", (&rows, &entries, false));
                for name in [rows::MODULE].into_iter().chain(packages) {
                    modules.del_item(name)?;
                }
                registered?;
                let price = rows
                    .get_item("lit-reentry")?
                    .expect("the registration landed")
                    .get_item("input_cost_per_token")?
                    .extract()?;
                Ok((host.getattr("found")?.extract()?, price))
            });
            sender
                .send(outcome.map_err(|error| error.to_string()))
                .unwrap();
        });

        let (found, price) = receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("register finished instead of deadlocking on its own lock")
            .unwrap();
        assert_eq!(found, [Some("Lit-Reentry".to_owned())]);
        assert!((price - 1.5).abs() < f64::EPSILON);
    }
}
