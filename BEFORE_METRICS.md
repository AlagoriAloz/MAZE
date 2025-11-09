## System State: 2024-01-15T10:30:00Z

### Performance (from last run)
- Total trades: 22
- Win rate: 27.3% (6W/16L)
- Cumulative PnL: -264 bps
- Last 10: 4W/6L (40%)
- Avg PnL/trade: -12 bps

### Models (current)
- Active: 11 (all voting)
- Weights: Momentum:20, Original:10, AntiOriginal:10, ...
- Training buffer: 4 samples

### Gist
- Size: ~1 MB
- Closed trades: 500+
- Pending: 10-20 (estimate)
- Equity entries: 1000+

### Issues to Fix
- ❌ 7 ML models undertrained (random predictions)
- ❌ AntiOriginal cancels Original (harmful)
- ❌ No regime switching
- ❌ Learning from garbage votes (buffer_size not tracked)
- ❌ Using ±1 weight updates (not scientific)
- ❌ No Gist trimming (1MB+ growth)
- ❌ No SAFE TRIM (risk of losing unprocessed trades)

### Expected Improvements
- Phase 1: 27% → 35% (remove harmful models)
- Phase 2-5: 35% → 45% (training gates + contamination fix)
- Phase 6-8: 45% → 58%+ (Wilson LCB + regime switching)
