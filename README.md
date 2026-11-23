# TimeNet Hackathon Round 1 Guide

Round 1 closes **Sunday, November 22, 2026 at 11:59 p.m. PST (UTC−8)**, which is **Monday, November 23 at 08:59 CET in Berlin**. Judging takes place November 23, and the submission form lists November 25 for the winner announcement. Later rounds will be announced in Discord. [Official challenge](https://challenge.timenet.ai/) · [Submission form](https://tally.so/r/J1qbyK)

Build a small application around a real time-series problem. Convert the dataset into TimeNet's TimeF format, train or fine-tune a time-series language model (TSLM), score it on held-out data and show what a person could do with the result. Classification, forecasting, anomaly detection and reasoning are all possible directions. [Official challenge](https://challenge.timenet.ai/)

## Join the hackathon

1. Form a team of **2–4 people**. Use Discord to find teammates or ask organizers questions. [TimeNet Discord](https://discord.gg/mqSkGPwuk)
2. Submit one registration for the team and include every member. Pick a main contact and save the exact team name and contact email for the submission form. Describe the problem, who it helps, the data source if known and how TimeNet will feed your model. [Team registration](https://tally.so/r/PRpLVe)
3. If you need compute, request a **$100 or $500 Nebius voucher** through registration. There is one voucher per team per round, subject to rolling review and availability. Participation is open with or without a voucher. The organizers also list AWS and Google Cloud credits for winning teams, subject to terms. [Official challenge FAQ](https://challenge.timenet.ai/#faq)

## Round 1 work plan

The plan below targets **Sunday, November 22 at 20:00 CET** for submission, with time left to fix links or upload issues.

| Berlin date | Milestone | Evidence to keep |
| --- | --- | --- |
| Nov 19 | Register, pick one task and dataset, start model setup | Source and license, task definition, checkpoint ID |
| Nov 20 | Build and reload TimeF, run a simple baseline and a short training job | Dataset version, split counts, first predictions, validation loss |
| Nov 21 | Finish training, freeze the configuration and score on held-out data. Publish the showcase posts early enough to collect engagement metrics | Baseline comparison, per-example predictions, one success and one failure |
| Nov 22 | Post in Discord, collect screenshots, verify public links and submit | Final showcase URL, social URLs, PR URL and submission confirmation |

Run connector and GPU setup in parallel if the team can split the work. The [build guide](BUILD.md) walks through TimeF, fine-tuning, evaluation and a worked example using office occupancy detection. Pick a different domain by replacing the dataset, task and score together.

## Submission checklist

- [ ] Prepare one entry for the team using the same team name and main contact email as registration.
- [ ] Write a short description covering the intended user, time-series task, TimeNet dataset, TSLM and held-out results. Include one limitation. Add source links and access instructions. For an existing project, identify what you added during the event.
- [ ] Link a public pitch, demo, website or other showcase. Include the project GitHub URL if useful. The repository field is optional.
- [ ] Include the TimeNet dataset PR URL if available. The challenge FAQ lists it in the submission package. The form introduction calls it optional, while a later section asks for it. Plan to include the PR. If you intend to submit without one, confirm with the organizers in Discord. [Challenge FAQ](https://challenge.timenet.ai/#faq) · [Submission form](https://tally.so/r/J1qbyK)
- [ ] Publish a project showcase post on **both X and LinkedIn**. Save both public URLs and screenshots of available post metrics, such as views or reactions, for judging. Make the screenshots accessible to judges and include them with the submission materials.
- [ ] Post the project in Discord **#showcase** before submitting. [TimeNet Discord](https://discord.gg/mqSkGPwuk)
- [ ] Open every link in a signed-out browser window and check that the team has permission to share the code, data and demo. Submit the form before the cutoff and save the confirmation. [Project submission](https://tally.so/r/J1qbyK)

Judges look at the use case and impact, technical work, dataset and use of TimeNet, quality of the evaluation, and the showcase and its public engagement. Use the demo to connect a real signal window, the model output and the decision it supports. Keep model scores distinct from social engagement metrics. [Official challenge FAQ](https://challenge.timenet.ai/#faq)

## Hill of Towie demo checkpoint

The selected checkpoint contains the update-1,500 fine-tuned weights for
`opentime/tslm-nov26` at revision `5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983`.
See the [held-out run report](checkpoints/hill-of-towie-demo/report.json).

With the 49 Git LFS objects available, run this from the repository root
to restore the weights to a new file. The restore script checks the original
SHA-256 before publishing the file.

```sh
python3 checkpoints/hill-of-towie-demo/restore.py /path/to/trained.safetensors
```
