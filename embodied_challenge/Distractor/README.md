# Distractor Asset Library

This folder stores candidate distractor assets for tableware tasks.

## Current status

- Total usable mesh assets: 25
- Main formats: .ply and .glb
- Category index: ASSET_INDEX.json

## Category summary

- PaperCup: 1
- plastic_bin: 1
- bottles: 10
- fork: 3
- mouse_pad: 1
- plate: 2
- spoon: 7

## Management rules

- Keep original source assets unchanged whenever possible.
- Add newly accepted assets into ASSET_INDEX.json first.
- Prefer mesh files with stable origin and scale.
- If a file is too large or unstable for physics, mark it in a future blacklist field in ASSET_INDEX.json.
- For task-level randomization, sample by category first, then sample inside category.

## Randomization recommendation for PourWater

- Suggested distractor count per reset: 2 to 5.
- Suggested categories: bottles, spoon, fork, plate, PaperCup, plastic_bin.
- Optional low probability category: mouse_pad.
- Keep a minimum safe distance from bottle and cup to avoid blocking success-critical interactions.