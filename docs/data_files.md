# Data files and running offline

Apart from four data files, everything the library needs is built in: the
EGM2008 gravity field to degree 360, the DE440 planetary ephemeris for
1950-2100, and leap seconds up to 2017. One call fetches the four files,
caches them, and refreshes them when they are out of date:

```python
import ballistic
ballistic.load()
```

| File | From | Size | Without it |
|------|------|------|------------|
| `finals2000A.data` or `.all` | IERS | ~3 MB | A propagation raises (accepting zeros instead, `set_eop_policy('assume_zero')`, puts frames 3-10 m off at low altitude) |
| `leap-seconds.list` | IERS / IETF | 5 KB | Leap seconds after 2017 unknown (there have been none so far) |
| `SW-All.csv` | CelesTrak | ~3 MB | Drag raises an error |
| `fes2004_Cnm-Snm.dat` | IERS Conventions | ~4 MB | No ocean tides, which together with the Step 2 solid tides reduce one-hour low-orbit error by 16-28% |

## Every external file and how it is loaded

| File | Source | Purpose | How it is loaded |
|------|--------|---------|-----------------|
| `finals2000A.data` | IERS | Polar motion and UT1-UTC; needed for sub-meter accuracy | `ballistic.load()`; found automatically afterwards |
| `leap-seconds.list` | IERS/IETF | Leap seconds after 2017 | `ballistic.load()`; found automatically afterwards |
| `SW-All.csv` | CelesTrak | Measured F10.7 and Ap since 1957, plus about 45 days of forecasts; drag raises an error without it | `ballistic.load()` or `ballistic.load(space_weather="path")`; C++ `SpaceWeatherData::instance().load_celestrak_csv(...)` |
| `fes2004_Cnm-Snm.dat` | IERS Conventions Ch. 6 | FES2004 ocean tides; third-party, so downloaded rather than shipped | `ballistic.load()`; C++ `gravity::OceanTideModel::instance().load(...)` |
| `de421.bsp` | JPL NAIF | ESA-compatible ephemeris, replacing the embedded DE440 | `jpl_ephemeris_path` |
| `EGM2008_to2190_TideFree` | NGA | Gravity degrees 361-2190 | `egm2008_path` |
| `nrl/ballistic_nrl.so` (`.dylib`, `.dll`) | this project's GitHub release | NRLMSIS 2.1 + HWM14 plugin for model-based drag; NRLMSIS 2.1 is for research, academic and non-profit use only | `ballistic.install_nrl_models(accept_license=True)` (not by `load()`); found automatically afterwards; C++ `nrl_models_path`, `BALLISTIC_NRL_MODELS` |

The embedded DE440 ephemeris (1950-2100) and EGM2008 to degree 360 cover
every orbit from LEO to HEO.

## Where the files are

`data_files()` shows which of these you already have and where they are:

```python
ballistic.data_files()
# {'eop': PosixPath('/home/you/.local/share/ballistic/finals2000A.data'),
#  'leap_seconds': PosixPath('.../leap-seconds.list'),
#  'space_weather': PosixPath('.../SW-All.csv'),
#  'ocean_tides': None}                      # None = not downloaded yet
```

`ballistic.doctor()` (or `python -m ballistic doctor`) goes further: it
loads the cached files, reports how far each one reaches, what is missing
whether the NRL drag models are installed, and runs a short self-test.

It looks in the default data directory (`ballistic.get_data_directory()`),
or in a folder you name: `ballistic.data_files("/path/to/kit")`.
`ballistic.Loader().status()` also reports each file's age.

## Running offline

An internet connection is not required if you already have the files. You
can name each file, and `offline=True` forbids downloads altogether:

```python
ballistic.load(eop="finals2000A.all", space_weather="SW-All.csv",
               ocean_tides="fes2004_Cnm-Snm.dat", offline=True)
```

`BALLISTIC_OFFLINE=1` in the environment makes every `ballistic.load()`
behave as `offline=True`, which suits air-gapped machines and CI.

To prepare an offline machine, download the files once on a connected one
into a folder you can carry (or copy the files that `data_files()` lists):

```python
ballistic.load(directory="/path/to/kit")      # downloads into /path/to/kit
```

On the offline machine, either point at the folder or name the files:

```python
ballistic.load(directory="/path/to/kit", offline=True)
ballistic.load(eop="/path/to/kit/finals2000A.data",
               leap_seconds="/path/to/kit/leap-seconds.list",
               space_weather="/path/to/kit/SW-All.csv",
               ocean_tides="/path/to/kit/fes2004_Cnm-Snm.dat")
```

In C++, pass the paths explicitly:

```cpp
InitConfig cfg;
cfg.eop_path = "/path/to/kit/finals2000A.data";
cfg.leap_seconds_path = "/path/to/kit/leap-seconds.list";
initialize(cfg);
atmosphere::SpaceWeatherData::instance().load_celestrak_csv("/path/to/kit/SW-All.csv");
gravity::OceanTideModel::instance().load("/path/to/kit/fes2004_Cnm-Snm.dat");
```

`initialize({})` finds `finals2000A.data` / `.all` and `leap-seconds.list`
in the data directory automatically; the space-weather and ocean-tide files
are loaded explicitly, as above.

The NRL drag models are not one of the data files: `load()` never fetches
them, because NRLMSIS 2.1 is licensed for non-commercial use only and is
installed only when you accept that. On a connected machine download the
plugin for the offline machine's platform from the
[GitHub release](https://github.com/nvhuynh16/ballistic/releases)
(`ballistic_nrl-linux-x86_64.so`, `-windows-x86_64.dll`, `-macos-arm64.dylib`
or `-macos-x86_64.dylib`; `ballistic.nrl_models().download_url` names the
right one), carry it over, and install it there:

```python
ballistic.install_nrl_models(accept_license=True, source="/path/to/kit/ballistic_nrl-linux-x86_64.so")
```

or copy it by hand to `ballistic.nrl_models().install_path` (the data
directory's `nrl/` folder, as `ballistic_nrl.so` / `.dylib` / `.dll`). In C++
set `cfg.nrl_models_path` or `BALLISTIC_NRL_MODELS`.

## How `load()` treats each file

`load()` decides each file separately, according to the argument you give
for it:

| Argument | `offline=False` (default) | `offline=True` |
|----------|---------------------------|----------------|
| a path | load that file, no download | same |
| `True` (default) | the copy in `directory`, downloaded if missing and refreshed if stale; if a download fails, the existing copy is used | the copy in `directory` as it is; skipped with a warning if there is none |
| `False` | skipped | skipped |

`offline` therefore matters only for files left at `True`, and it is fine to
name some files and not others.

The loaded data stays in memory for the life of the process, and each
successful load replaces it. A load that fails keeps the data already in
memory and says so. Because downloads are stored in the same directory that
`offline=True` reads, switching to offline mode after an online session
picks up the files that were just downloaded.

## Keeping the files fresh

For historical work, any copy of the files that covers your dates is
enough. For current work, keep the files fresh: the space-weather
measurements stop on the download date (after which drag runs on forecasts
for about 45 days, and then stops), and the Earth orientation file runs
about a year past its download date. Beyond the Earth orientation table the
edge value is held for up to 30 days with a warning, and then the
propagation stops. What happens at the edge of every table is listed in
[`out_of_range_policy.md`](out_of_range_policy.md).
