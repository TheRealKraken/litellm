use std::borrow::Cow;
use std::collections::HashMap;

use indexmap::IndexSet;
use litellm_host_python::to_py;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{CostMap, Written};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple};
use serde_json::Value;

pub(crate) const MODULE: &str = "litellm.rust_bridge.host.model_cost";
const FALLBACK_GENERALIZATIONS: &str = "fallback_generalizations";
const ALIASES: &str = "aliases";

/// What one key of `litellm.model_cost` holds
enum Served<'a> {
    Model,
    /// The very object served under the owner
    AliasOf(&'a str),
    /// A top-level value that is not a model row, such as `sample_spec`, served as written
    Section(&'a Value),
}

/// The keys main's `_expand_model_aliases` left in `litellm.model_cost`, in its dict order: every top-level key but
/// `fallback_generalizations` in document order, then each owner's listed aliases. An `alias_of` row whose target
/// is not a model is dropped. An `alias_of` row repeats among the aliases, and keeps its document position
fn served(cost_map: &CostMap) -> impl Iterator<Item = (&str, Served<'_>)> {
    let catalog = cost_map.catalog();
    let documented = catalog
        .written()
        .filter(|(key, _)| *key != FALLBACK_GENERALIZATIONS)
        .filter_map(|(key, written)| {
            let served = match written {
                Written::Section(value) => Served::Section(value),
                Written::Entry(_) => match catalog.aliases().get(key) {
                    Some(owner) => Served::AliasOf(owner),
                    None => catalog.get(key).map(|_| Served::Model)?,
                },
            };
            Some((key, served))
        });
    let listed = cost_map.alias_groups().flat_map(|(owner, names)| {
        names
            .iter()
            .skip(1)
            .map(move |alias| (alias.as_str(), Served::AliasOf(owner)))
    });
    documented.chain(listed)
}

/// The installed document as `litellm.model_cost`, every alias sharing its owner's object
pub(crate) fn table<'py>(py: Python<'py>, cost_map: &CostMap) -> PyResult<Bound<'py, PyDict>> {
    let table = PyDict::new(py);
    let mut models = HashMap::new();
    for (key, served) in served(cost_map) {
        let value = match served {
            Served::Model => model(py, cost_map, key, &mut models)?,
            Served::AliasOf(owner) => model(py, cost_map, owner, &mut models)?,
            Served::Section(value) => to_py(py, value)?.into_bound(py),
        };
        table.set_item(key, value)?;
    }
    Ok(table)
}

fn model<'a, 'py>(
    py: Python<'py>,
    cost_map: &'a CostMap,
    key: &'a str,
    models: &mut HashMap<&'a str, Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(row) = models.get(key) {
        return Ok(row.clone());
    }
    let fields = cost_map
        .catalog()
        .get(key)
        .expect("a served model or alias owner is a catalog row");
    let fields = to_py(py, without_alias_list(fields).as_ref())?.into_bound(py);
    let names = cost_map.names(key);
    let row = match names.is_empty() {
        true => fields,
        false => py.import(MODULE)?.getattr("entry")?.call1((
            fields,
            PyTuple::new(py, names)?,
            PyFrozenSet::empty(py)?,
        ))?,
    };
    models.insert(key, row.clone());
    Ok(row)
}

/// A row as main served it, without the `aliases` list its document carried
fn without_alias_list(fields: &Fields) -> Cow<'_, Fields> {
    if fields.get(ALIASES).is_none_or(Value::is_null) {
        return Cow::Borrowed(fields);
    }
    Cow::Owned(
        fields
            .iter()
            .filter(|(name, _)| *name != ALIASES)
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    )
}

/// The installed row under `key` with `applied` written over it in order, `None` when neither has one
pub(crate) fn registered_row<'py>(
    py: Python<'py>,
    cost_map: &CostMap,
    key: &str,
    applied: &[Bound<'py, PyDict>],
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let catalog = cost_map.catalog();
    let installed = catalog
        .get(key)
        .or_else(|| match catalog.section(key) {
            Some(Value::Object(fields)) if key != FALLBACK_GENERALIZATIONS => Some(fields),
            _ => None,
        })
        .map(without_alias_list);
    if installed.is_none() && applied.is_empty() {
        return Ok(None);
    }
    let row = match installed {
        Some(fields) => to_py(py, fields.as_ref())?
            .into_bound(py)
            .cast_into::<PyDict>()?,
        None => PyDict::new(py),
    };
    for registration in applied {
        row.update(registration.as_mapping())?;
    }
    Ok(Some(row))
}

/// The keys `registered_row` answers for, installed keys first
pub(crate) fn registered_keys<'a>(
    cost_map: &'a CostMap,
    registered: impl Iterator<Item = &'a str>,
) -> Vec<String> {
    let served: IndexSet<&str> = served(cost_map).map(|(key, _)| key).collect();
    let only_registered: Vec<&str> = registered.filter(|key| !served.contains(key)).collect();
    served
        .into_iter()
        .chain(only_registered)
        .map(str::to_owned)
        .collect()
}

/// Where a registration lands in a host's table and the row it starts from
pub(crate) struct Landing<'py> {
    pub(crate) key: String,
    pub(crate) base: Bound<'py, PyAny>,
    pub(crate) matched_builtin: bool,
}

/// Where `registration` lands in `rows`, or `None` when `rows` is not the table the host publishes, since the host
/// looks keys up only in that one
pub(crate) fn resolve<'py>(
    rows: &Bound<'py, PyDict>,
    key: &str,
    registration: &Bound<'py, PyAny>,
    custom_llm_provider: Option<&str>,
) -> PyResult<Option<Landing<'py>>> {
    let landing = rows.py().import(MODULE)?.getattr("resolve")?.call1((
        rows,
        key,
        registration,
        custom_llm_provider,
    ))?;
    if landing.is_none() {
        return Ok(None);
    }
    Ok(Some(Landing {
        key: landing.getattr("key")?.extract()?,
        base: landing.getattr("base")?,
        matched_builtin: landing.getattr("matched_builtin")?.extract()?,
    }))
}

/// Writes `registration` over the row under `landing.key`, and under every alias name that shares that row object
pub(crate) fn write<'py>(
    rows: &Bound<'py, PyDict>,
    landing: &Landing<'py>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let current = rows.get_item(&landing.key)?;
    let row = rows.py().import(MODULE)?.getattr("merged")?.call1((
        current.as_ref(),
        &landing.base,
        registration,
    ))?;
    let names: Vec<String> = row.getattr("names")?.extract()?;
    let sharing: Vec<String> = names
        .into_iter()
        .filter(|name| {
            rows.get_item(name)
                .ok()
                .flatten()
                .is_some_and(|served| current.as_ref().is_some_and(|current| served.is(current)))
        })
        .collect();
    rows.set_item(&landing.key, &row)?;
    sharing
        .iter()
        .try_for_each(|name| rows.set_item(name, &row))
}

/// The registration kept for a key: `registration` merged into the one kept before it
pub(crate) fn kept<'py>(
    previous: Option<&Bound<'py, PyDict>>,
    registration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(registration
        .py()
        .import(MODULE)?
        .getattr("kept")?
        .call1((previous, registration))?
        .cast_into::<PyDict>()?)
}

/// Tells the host a write landed in its table, so it drops what it derived from the table
pub(crate) fn changed(py: Python<'_>) -> PyResult<()> {
    py.import(MODULE)?.getattr("changed")?.call0().map(drop)
}
