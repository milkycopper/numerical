rustup component add llvm-tools-preview
$env:RUSTFLAGS="-Cinstrument-coverage"
cargo build
$env:LLVM_PROFILE_FILE="target/profraw/bo-%p-%m.profraw"
cargo test
grcov . -s . --binary-path ./target/debug/ -t html --branch --ignore-not-existing -o ./target/debug/coverage/
Invoke-Expression ./target/debug/coverage/html/index.html