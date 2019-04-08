# plenopy

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Read, plot, and investigate the recorded events of the Atmospheric Cherenkov Plenoscope (ACP). Plenopy reads the ACP events written by the Cherenkov-plenoscope-simulation in [merlict](https://github.com/cherenkov-plenoscope/merlict_development_kit).

## install
```bash
pip install git+https://github.com/cherenkov-plenoscope/plenopy/
```


## basic usage
```python
import plenopy as pl

run = pl.Run('plenopy/plenopy/tests/resources/run.acp/')
event = run[3]
event.show()
```
![img](readme/example_event_show.png)
