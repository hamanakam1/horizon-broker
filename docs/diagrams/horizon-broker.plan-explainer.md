# How should the Horizon Broker be built from here? {#plan}

The next release is a correctness release, not a feature release: **close the defects that make the behaviour and the documentation disagree** before taking on more work. The Phase 0 defects below are now implemented; the remaining plan focuses on measured throughput and multi-broker control.

- A quarantined node returns to the pool after its configured cooling-off period.
- Terminal completion, failure, cancellation and expiry resolve waiting HTTP clients.
- Completion and failure counters, quantiles and batch metrics are exported from observed lifecycle events.
- The Flex planner groups work, while real multi-prompt admission is limited to opt-in agent-container nodes; direct Ollama remains one request per prompt.

@answer scope="the plan from v0.3 to v1.0 · one broker, one host, one fleet today"
@meta locale=en style=editorial repo_url=https://github.com/hamanakam1/horizon-broker repo_revision=76e545fd9e6ed8ce58dc74916f245979154941da
@source plan title="The long-term plan" path=docs/ROADMAP.md lines=60:100
@source quarantine title="The quarantine check" path=horizon_broker/routing/placement.py lines=45:56
@source waiter title="The terminal paths of a job" path=horizon_broker/scheduler/engine.py lines=240:300
@source counters title="The declared metrics" path=horizon_broker/scheduler/engine.py lines=70:95
@source flexbatch title="The Flex dispatch" path=horizon_broker/scheduler/engine.py lines=395:410
@source budget title="The recovery budget" path=docs/RUNBOOK.md lines=160:175

## What each phase has to prove {#phases}

@workflow id=delivery-plan title="The delivery plan"
| lane build | Build |
| lane gate | Release gates |
| lane release | Release |
| truth | Phase 0 · Truth | backend | build |
| gateA | Gate A | security | gate |
| speed | Phase 1 · Speed | backend | build |
| gateB | Gate B | security | gate |
| scale | Phase 2 · Scale | backend | build |
| v1 | v1.0 | security | release |
| truth > gateA | defects closed |
| gateA > speed | green | emphasis |
| speed > gateB | evidence attached |
| gateB > scale | soak passed | emphasis |
| scale > v1 | SLOs held | emphasis |

@steps id=plan-steps
| truth | Phase 0 · Truth | Close the defects that make the behaviour and the documentation disagree. | delivery-plan | truth |
| gateA | Gate A | The fault-path suite must be green, not only the unit suite. | delivery-plan | gateA |
| speed | Phase 1 · Speed | Make the batching promise real and survive a broker restart. | delivery-plan | speed |
| gateB | Gate B | A week of soak evidence with measured numbers. | delivery-plan | gateB |
| scale | Phase 2 · Scale | Two brokers, one fleet, one job. | delivery-plan | scale |
| v1 | v1.0 | Seven days of fault injection on the real fleet. | delivery-plan | v1 |

## What each phase buys and what it costs {#comparison}

| Phase | Version | Closes | Gate to ship |
| --- | --- | --- | --- |
| Truth | v0.3 | The quarantine, the draining health, the waiters, the counters | The fault-path suite is green |
| Speed | v0.4 | The fake batch, the fake stream, the unreachable journal | A week of soak with measured MFU |
| Scale | v0.5 | Two brokers, one fleet, one job | The leader-election drill |
| Proof | v1.0 | Support, runbooks, signed releases | Seven days of fault injection |

@facts id=numbers
| Tests in the suite | 275 |
| Modules in the package | 42 |
| Diagram types published | 10 |
| Recovery budget | 30 s |

@metric id=recovery-budget label="Recovery budget" value=30 unit=s note="the outer bound the plan promises, from the runbook"

## The one item that is a decision, not a defect {#decision}

@callout id=control-plane level=open-question title="Is one broker enough?"
| Two brokers on one network both accept and run the same job id today. The plan fixes that with a control plane, but only after the correctness work lands. If the fleet grows past one host before Phase 2, the answer is leader election first and throughput later.

## What to do first {#checklist}

- [x] Write the failing test for quarantine expiry, then fix the check.
- [x] Resolve the waiting future on failure, cancellation and expiry paths.
- [x] Increment lifecycle counters and declare recovery quantiles.
- [ ] Make direct-engine Flex execution call a real multi-prompt endpoint; opt-in agent containers already receive one batch assignment.
- [ ] Publish a week of soak numbers before promising a throughput gain.
