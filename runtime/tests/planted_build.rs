//! Planted bugs are compile-time only.
//!
//! They exist only in a build with the `planted-bugs` Cargo feature (off by
//! default); without it there is no field, setter or flag that could switch
//! one on. A `planted-bugs` build refuses to write recorded results or
//! traces, so no artifact can come from it.

#[cfg(not(feature = "planted-bugs"))]
#[test]
fn default_build_has_no_planted_bugs() {
    // Checked at compile time.
    const { assert!(!fp_play_runtime::PLANTED_BUGS) };
}

#[cfg(feature = "planted-bugs")]
#[test]
fn planted_build_refuses_to_record() {
    const { assert!(fp_play_runtime::PLANTED_BUGS) };
    let out = std::env::temp_dir().join(format!("fp-play-planted-{}", std::process::id()));
    for command in ["experiment", "traces"] {
        let run = std::process::Command::new(env!("CARGO_BIN_EXE_fp-play-runtime"))
            .args([command, "--out"])
            .arg(&out)
            .output()
            .expect("binary runs");
        assert!(!run.status.success(), "{command} ran in a planted-bugs build");
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(stderr.contains("planted-bugs"), "{stderr}");
    }
    assert!(!out.exists(), "a planted-bugs build wrote {}", out.display());
}
