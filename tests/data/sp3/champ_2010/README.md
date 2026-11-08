# CHAMP final weeks (2010)

GFZ Rapid Science Orbits (RSO, v01) for CHAMP's last available days,
2010 days 236-247 (24 August to 4 September), at 249-283 km. Used by
`scripts/validate_champ_drag.py` for the `CHAMP-260KM` row. CHAMP re-entered
on 19 September 2010; GFZ's RSO series ends with these files.

- Files: `GFZOP_RSO_L06_G_*_v01.sp3.gz`, verbatim, two overlapping arcs per
  day (10:00-24:00 and 22:00-12:00 next day).
- Format: SP3, 30 s positions and velocities, GPS time, Earth-fixed ("CTS";
  treated as ITRF2005, as for the 2005 CHAMP files).
- Source: GFZ ISDC, `https://isdc-data.gfz.de/champ/ORBIT/L06/RSO/2010/<doy>/`
  (no registration). DOI: https://doi.org/10.5880/GFZ_ORBIT/RSO/L06_G_v01
- Downloaded 2026-10-31.

To reacquire:

```bash
for d in $(seq 236 247); do
  for f in $(curl -s https://isdc-data.gfz.de/champ/ORBIT/L06/RSO/2010/$d/ | grep -o 'GFZOP_RSO_[^"]*\.sp3\.gz' | sort -u); do
    curl -O https://isdc-data.gfz.de/champ/ORBIT/L06/RSO/2010/$d/$f
  done
done
```
