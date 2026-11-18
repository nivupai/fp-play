//! Golden serving traces (tests/traces/*.jsonl), compared byte for byte.

use fp_play_runtime::serving::engine::run;
use fp_play_runtime::serving::traces::{golden, three_requests, to_jsonl, LINEAR};

#[test]
fn golden_traces_are_current() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/traces");
    for (name, contents) in golden().unwrap() {
        let recorded = std::fs::read_to_string(format!("{dir}/{name}"))
            .unwrap_or_else(|_| panic!("tests/traces/{name} missing: run `cargo run --release -- traces`"));
        assert!(
            contents == recorded,
            "tests/traces/{name} differs; if the change is intended, run `cargo run --release -- traces` and review the diff"
        );
    }
}

#[test]
fn hand_derived_trace_lines() {
    let (requests, config) = three_requests();
    let text = to_jsonl(&run(&requests, &config, &LINEAR).unwrap().steps);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4);
    assert_eq!(
        lines[1],
        r#"{"step":1,"start_ns":150,"end_ns":310,"swapped_in":[],"admitted":[1,2],"preempted":[],"stalls":0,"work":[[0,0,5,1],[1,0,0,3],[2,0,0,2]],"cow_blocks":0,"swap_blocks":0,"commits":[[0,0,1],[1,0,0],[1,1,0]],"finished":[0],"used_blocks":2}"#
    );
}
