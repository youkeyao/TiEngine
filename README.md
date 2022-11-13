# TiEngine

Several physical simulation systems implememted by taichi.

- Mass-Spring
- SPH Fluid
    - WCSPH
    - PCISPH

## Mass-Spring
```
configs = {
    "title": "MassSpring",
    "model": MassSpring,
    "type": 2,
    "dt": 1e-3,
    "t": 8
}
```
![Mass-Spring](figs/massspring.gif)

## WCSPH
```
configs = {
    "title": "WCSPH",
    "model": SPHFluid,
    "type": 1,
    "dt": 1e-3,
    "t": 4
}
```
![WCSPH](figs/wcsph.gif)

## PCISPH
configs = {
    "title": "PCISPH",
    "model": SPHFluid,
    "type": 2,
    "dt": 1e-3,
    "t": 4
}
![PCISPH](figs/pcisph.gif)