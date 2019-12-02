To compile a stage, use `make STAGE=n` where `n` is the stage number. For example, `make STAGE=3`.

| Stage | Description                    |
|:-----:|--------------------------------|
|   0   | CPU calculation (default code) |
|   1   | GPU with naive offloading      |
|   2   | Optimized data movement        |
|   3   | Establishing lifetime          |
