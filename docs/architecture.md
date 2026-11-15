1. Status/Result，不要用 bool 同时表示 EOF 和错误。
2. 所有异步接口必须显式带 cudaStream_t，禁止隐式 default stream（我们已经在 TRT 测试里看到 default stream 的告警）。
3. 谁分配、谁持有、谁复用要有明确 owner；HostFrame 在入队后不能被解码线程覆盖。