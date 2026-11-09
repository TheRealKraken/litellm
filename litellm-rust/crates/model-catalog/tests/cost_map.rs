use litellm_model_catalog::{
    BUNDLED_REGISTRY, CostMap, ErrorKind, IntegrityLimits, Provenance, Refused, git_blob_id,
};
use rstest::rstest;
use serde_json::json;

fn any_size() -> IntegrityLimits {
    IntegrityLimits::new(0, 0.0).unwrap()
}

#[rstest]
#[case::not_json(b"{".as_slice(), any_size(), false, ErrorKind::Json)]
#[case::empty(b"{}".as_slice(), any_size(), true, ErrorKind::Empty)]
#[case::below_minimum(br#"{"m":{"litellm_provider":"p"}}"#.as_slice(), IntegrityLimits::new(2, 0.0).unwrap(), true, ErrorKind::BelowMinimum)]
#[case::shrunk(br#"{"m":{"litellm_provider":"p"}}"#.as_slice(), IntegrityLimits::new(0, 0.5).unwrap(), true, ErrorKind::Shrunk)]
fn a_refused_document_says_whether_it_failed_the_integrity_limits(
    #[case] body: &[u8],
    #[case] limits: IntegrityLimits,
    #[case] integrity: bool,
    #[case] kind: ErrorKind,
) {
    let error = match CostMap::read(body, None, limits).unwrap_err() {
        Refused::Integrity(error) if integrity => error,
        Refused::Unreadable(error) if !integrity => error,
        refused => panic!("{refused:?}"),
    };
    assert_eq!(error.kind(), kind);
}

#[rstest]
fn provenance_is_the_read_bytes_and_etag() {
    let body = br#"{"m":{"litellm_provider":"p"}}"#;
    let cost_map = CostMap::read(body, Some("\"v1\"".into()), any_size()).unwrap();
    assert_eq!(
        cost_map.provenance(),
        &Provenance {
            revision: "eaa747f210a1527d5ea7138be0f768c47578ca68".to_owned(),
            etag: Some("\"v1\"".into()),
        }
    );
    assert_eq!(
        CostMap::bundled().provenance().revision,
        git_blob_id(BUNDLED_REGISTRY)
    );
}

#[rstest]
fn every_name_of_an_aliased_row_lists_its_owner_first() {
    let body = json!({
        "owner": {"litellm_provider": "p", "aliases": ["short"]},
        "owner-latest": {"alias_of": "owner"},
        "plain": {"litellm_provider": "p"},
    })
    .to_string();
    let cost_map = CostMap::read(body.as_bytes(), None, any_size()).unwrap();
    for key in ["owner", "short", "owner-latest"] {
        assert_eq!(cost_map.names(key), ["owner", "short", "owner-latest"]);
    }
    assert!(cost_map.names("plain").is_empty());
}
