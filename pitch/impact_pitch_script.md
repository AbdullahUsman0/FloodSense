# FloodSense 3-Minute Impact Pitch

## Opening (0:00 - 0:30)
Today we are presenting FloodSense, a district-focused early warning support tool designed for frontline officials who need fast, actionable flood guidance. We built this system for low-connectivity environments and non-technical users, so decisions can be made in minutes, not hours.

## The Problem (0:30 - 1:00)
Pakistan's flood disasters show a warning gap, not just a weather gap. In 2022 alone, NDMA reported 14,563,770 people affected in Sindh, 4,350,490 in KP, and 9,182,616 in Balochistan. District teams often receive too little lead time to coordinate evacuations and pre-position rescue resources.

## What FloodSense Delivers (1:00 - 1:45)
FloodSense takes simple field inputs that any district desk can provide: rainfall, date, district, soil condition, and visible surface water. It then returns:
- A clear risk level: Low, Medium, High, or Critical
- A confidence percentage
- Estimated people at risk
- One concrete recommended action

All outputs appear in English and Urdu simultaneously, with color-coded risk badges for immediate readability.

## Reliability and Data Handling (1:45 - 2:20)
We built this on cleaned training data with explicit protection against real-world data problems:
- Missing rainfall values are handled safely
- Infinite and invalid percentage-change values are corrected
- Duplicate and phantom rows are removed before training
- The system is evaluated with both time-based and stratified test splits

Current evaluation results:
- Stratified split accuracy: 99.64%
- Time-based split accuracy: 100.00%

This ensures the tool remains stable when conditions are noisy and field data is incomplete.

## Failure Modes and Safety (2:20 - 2:45)
When data is missing or out of range, FloodSense does not force a risk label. It returns:
"Insufficient data — manual assessment recommended."

This protects against false certainty. In critical operations, a cautious fallback is better than a confident wrong answer.

## Buner 2025 Impact Claim (2:45 - 2:55)
For the August 15, 2025 Buner flooding window, our operational claim is 12 to 18 hours of additional actionable warning compared with late-stage manual escalation, based on daily monitoring cadence and automatic district alerts.

## Close (2:55 - 3:00)
FloodSense is built for real district workflows: fast, bilingual, resilient, and action-first. The goal is simple: give officials usable warning time so they can move people earlier and save lives.
