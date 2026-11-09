use std::collections::BTreeSet;

use indexmap::IndexMap;

/// `register_model(persist_across_reloads=True)` calls, replayed above every installed document. `V` is the host's
/// own registration value, merged by the host, so a value the host holds never has to cross as JSON
#[derive(Clone, Debug, PartialEq)]
pub struct Registrations<V> {
    /// Each registered key's merged value, ordered by its latest registration
    entries: IndexMap<String, V>,
    /// The registered keys whose write landed on each row key, in landing order
    landed_on: IndexMap<String, Vec<String>>,
}

impl<V> Default for Registrations<V> {
    fn default() -> Self {
        Self {
            entries: IndexMap::new(),
            landed_on: IndexMap::new(),
        }
    }
}

/// One registration written to a host's table: the key it was registered under and the row key it landed on
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Landed {
    pub registered: String,
    pub written: String,
}

impl<V> Registrations<V> {
    /// Keeps `value` as the key's registration, already merged with `get(key)` by the host, and makes it the latest
    pub fn register(&mut self, key: &str, value: V) {
        self.entries.shift_remove(key);
        self.entries.insert(key.to_owned(), value);
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.entries.get(key)
    }

    pub fn record(&mut self, landed: &[Landed]) {
        for Landed {
            registered,
            written,
        } in landed
        {
            let names = self.landed_on.entry(written.clone()).or_default();
            if !names.contains(registered) {
                names.push(registered.clone());
            }
        }
    }

    /// Every registration, oldest first
    pub fn entries(&self) -> impl Iterator<Item = (&str, &V)> {
        self.entries
            .iter()
            .map(|(key, value)| (key.as_str(), value))
    }

    /// The registrations that apply to the row under any of `names`, oldest first, as `_RegistrationsOverCatalog` reads them
    pub fn over<'a>(&'a self, names: &[&str]) -> Vec<&'a V> {
        let mut applied: Vec<(usize, &V)> = names
            .iter()
            .flat_map(|name| {
                self.landed_on
                    .get(*name)
                    .into_iter()
                    .flatten()
                    .map(String::as_str)
                    .chain(std::iter::once(*name))
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|key| self.entries.get_full(key))
            .map(|(order, _, value)| (order, value))
            .collect();
        applied.sort_unstable_by_key(|(order, _)| *order);
        applied.into_iter().map(|(_, value)| value).collect()
    }

    pub fn map<W>(&self, mut convert: impl FnMut(&V) -> W) -> Registrations<W> {
        Registrations {
            entries: self
                .entries
                .iter()
                .map(|(key, value)| (key.clone(), convert(value)))
                .collect(),
            landed_on: self.landed_on.clone(),
        }
    }
}

/// A host's live table that registrations are written into
pub trait RegistrationTable {
    type Registration;
    type Landing;
    type Error;

    /// Where `registration` lands, `None` when this table is no longer the one the host publishes
    fn resolve(
        &mut self,
        key: &str,
        registration: &Self::Registration,
    ) -> Result<Option<Self::Landing>, Self::Error>;

    fn write(
        &mut self,
        landing: Self::Landing,
        registration: &Self::Registration,
    ) -> Result<Placed, Self::Error>;
}

/// Where a registration was written and whether a built-in row matched it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub key: String,
    pub matched_builtin: bool,
}

#[derive(Debug, PartialEq)]
pub enum Pass<'a, R> {
    /// Every entry was written. `unmatched` are those no built-in row matched
    Finished {
        landed: Vec<Landed>,
        unmatched: Vec<(&'a str, &'a R)>,
    },
    /// The table stopped being the published one before every entry was written
    Swapped { landed: Vec<Landed> },
}

impl<R> Pass<'_, R> {
    pub fn landed(&self) -> &[Landed] {
        match self {
            Pass::Finished { landed, .. } | Pass::Swapped { landed } => landed,
        }
    }
}

/// Writes `entries` into `table` in order, each resolved after the writes before it, so a lookup sees them
pub fn write_pass<'a, T: RegistrationTable>(
    table: &mut T,
    entries: &'a [(String, T::Registration)],
) -> Result<Pass<'a, T::Registration>, T::Error> {
    let mut landed = Vec::with_capacity(entries.len());
    let mut unmatched = Vec::new();
    for (key, registration) in entries {
        let Some(landing) = table.resolve(key, registration)? else {
            return Ok(Pass::Swapped { landed });
        };
        let written = table.write(landing, registration)?;
        if !written.matched_builtin {
            unmatched.push((key.as_str(), registration));
        }
        landed.push(Landed {
            registered: key.clone(),
            written: written.key,
        });
    }
    Ok(Pass::Finished { landed, unmatched })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn landed(registered: &str, written: &str) -> Landed {
        Landed {
            registered: registered.to_owned(),
            written: written.to_owned(),
        }
    }

    #[rstest]
    fn reregistering_a_key_makes_it_the_latest() {
        let mut registrations = Registrations::default();
        registrations.register("a", 1);
        registrations.register("b", 2);
        registrations.register("a", 3);
        assert_eq!(
            registrations.entries().collect::<Vec<_>>(),
            [("b", &2), ("a", &3)]
        );
    }

    #[rstest]
    fn over_applies_names_landed_on_the_row_in_registration_order() {
        let mut registrations = Registrations::default();
        registrations.register("openai/o3-pro", 1);
        registrations.register("o3-pro", 2);
        registrations.register("other", 3);
        registrations.record(&[
            landed("openai/o3-pro", "o3-pro"),
            landed("o3-pro", "o3-pro"),
        ]);
        registrations.register("openai/o3-pro", 4);
        assert_eq!(registrations.over(&["o3-pro"]), [&2, &4]);
    }

    /// A table published until `swaps_after` writes, whose rows match a built-in only when the key starts with `b`
    struct Table {
        writes: Vec<(String, i32)>,
        swaps_after: usize,
    }

    impl RegistrationTable for Table {
        type Registration = i32;
        type Landing = String;
        type Error = ();

        fn resolve(&mut self, key: &str, _: &i32) -> Result<Option<String>, ()> {
            Ok((self.writes.len() < self.swaps_after).then(|| key.to_uppercase()))
        }

        fn write(&mut self, landing: String, registration: &i32) -> Result<Placed, ()> {
            self.writes.push((landing.clone(), *registration));
            Ok(Placed {
                matched_builtin: landing.starts_with('B'),
                key: landing,
            })
        }
    }

    #[rstest]
    fn a_pass_writes_every_entry_in_order_and_reports_the_unmatched() {
        let entries = [("a".to_owned(), 1), ("b".to_owned(), 2)];
        let mut table = Table {
            writes: Vec::new(),
            swaps_after: usize::MAX,
        };
        assert_eq!(
            write_pass(&mut table, &entries),
            Ok(Pass::Finished {
                landed: vec![landed("a", "A"), landed("b", "B")],
                unmatched: vec![("a", &1)],
            })
        );
        assert_eq!(table.writes, [("A".to_owned(), 1), ("B".to_owned(), 2)]);
    }

    #[rstest]
    fn a_pass_stops_at_the_first_entry_resolved_after_a_swap() {
        let entries = [("a".to_owned(), 1), ("b".to_owned(), 2)];
        let mut table = Table {
            writes: Vec::new(),
            swaps_after: 1,
        };
        assert_eq!(
            write_pass(&mut table, &entries),
            Ok(Pass::Swapped {
                landed: vec![landed("a", "A")],
            })
        );
        assert_eq!(table.writes, [("A".to_owned(), 1)]);
    }
}
