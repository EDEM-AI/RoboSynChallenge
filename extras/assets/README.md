# extras/assets

Extra object meshes for the RoboSynChallenge simulation, used together with the rest of `extras/` (shape generators, environment patches, anomaly checks). Everything here is included in the pull request.

Each `items/<name>/` folder holds the meshes (OBJ / PLY), `*.annotations.json` (functional parts) where available, and a `variant.yaml` with source, licence and sha256 of every file. Some items also ship URDF or collision pieces.

| item | class | origin | licence | author |
|---|---|---|---|---|
| `bell_02_77f2ea_84` | bell | third-party | CC BY 4.0 | gla_bot |
| `bell_03_b739cb_93` | bell | third-party | CC BY 4.0 | davide.6.6.95 |
| `fork_01_7540e0_175` | fork | third-party | CC BY 4.0 | azureinferno |
| `fork_02_780905_183` | fork | third-party | CC BY 4.0 | BrisM |
| `spoon_01_2cdb7b_135` | spoon | third-party | CC BY 4.0 | quedlin |
| `spoon_03_590860_153` | spoon | third-party | CC BY 4.0 | rastakouere.olivier |
| `plate_03_57ada5_195` | plate | third-party | CC BY 4.0 | GfS-Rostock |
| `plate_04_6f7201_210` | plate | third-party | CC0 1.0 | artsmia |
| `sharpie_mid_15` | marker | third-party | CC BY 4.0 | adim09 |
| `sharpie_thick_20` | marker | third-party | CC BY 4.0 | adim09 |
| `milk_flat_cap` | milk | third-party | CC BY 4.0 | hardbottle |
| `milk_gable_nocap` | milk | third-party | CC BY 4.0 | cebbi |
| `rack_4x10_main_zc` | rack | own | CC BY 4.0 | Yi Li (liyifreddy) |
| `rack_4x12_main_zc` | rack | own | CC BY 4.0 | Yi Li (liyifreddy) |
| `rack_5x10_tall_zc` | rack | own | CC BY 4.0 | Yi Li (liyifreddy) |
| `holder_bore86_h96_yup_solid` | holder | own | CC BY 4.0 | Yi Li (liyifreddy) |
| `drawer_handle_shallow_arc` | drawer_handle | own | CC BY 4.0 | Yi Li (liyifreddy) |
| `drawer_handle_bridge` | drawer_handle | own | CC BY 4.0 | Yi Li (liyifreddy) |

## Use

Point the tools at `extras/assets/items/<name>/`. `extras/assets_manifest.json` lists every file with its sha256.
