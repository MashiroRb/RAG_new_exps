# Experiment Summary (auto-generated)

## E0 calibration (camera)
- rho_hat = 0.3336, R2 = 0.9714, MAE = 0.0317, RMSE = 0.0386 (n=87)

## E0 calibration (ev)
- rho_hat = 0.9188, R2 = 0.9669, MAE = 0.0223, RMSE = 0.028 (n=87)

## E0 calibration (shoes)
- rho_hat = 0.8391, R2 = 0.9579, MAE = 0.0231, RMSE = 0.0311 (n=87)

## E1 filtering response
- mean |q-meas − model|: fixed=0.0301, adaptive=0.0528
- strategic offset q_adapt − q_fixed by r: r=0.00:+0.211, r=0.20:+0.233, r=0.40:+0.258, r=0.60:+0.203, r=0.80:+0.041, r=0.95:+0.027

## E2 authentic-mass lever
- abs80: rho_implied by corpus fraction {'0.25': 0.0534, '0.5': 0.0451, '1.0': 0.0369}, ratio to full {'0.25': 1.4472, '0.5': 1.2222, '1.0': 1.0}, rho/f constancy {'0.25': 0.2137, '0.5': 0.0902, '1.0': 0.0369}
- scaled: rho_implied by corpus fraction {'0.25': 0.0884, '0.5': 0.0626, '1.0': 0.0369}, ratio to full {'0.25': 2.3957, '0.5': 1.6965, '1.0': 1.0}, rho/f constancy {'0.25': 0.3535, '0.5': 0.1251, '1.0': 0.0369}

## E3 N-comparative statics (equilibrium arms, r=0)
- equilibrium_w* N=1: x*/provider=32.0 docs, total=32.0 docs, q_meas=0.380 vs q0=0.356
- equilibrium_w* N=2: x*/provider=29.0 docs, total=58.0 docs, q_meas=0.532 vs q0=0.500
- equilibrium_w* N=4: x*/provider=25.0 docs, total=100.0 docs, q_meas=0.657 vs q0=0.627
- equilibrium_w* N=8: x*/provider=19.0 docs, total=152.0 docs, q_meas=0.786 vs q0=0.726
- equilibrium_w1 N=1: x*/provider=86.0 docs, total=86.0 docs, q_meas=0.636 vs q0=0.595
- equilibrium_w1 N=2: x*/provider=85.0 docs, total=170.0 docs, q_meas=0.814 vs q0=0.743
- omega sweep (N=2, cap-clamped where x* exceeds pool), 3-seed means:
  - omega=0.5: x*=59 q_meas=0.728 vs q0=0.666
  - omega=1.0: x*=85 q_meas=0.814 vs q0=0.743
  - omega=2.0: x*=122 q_meas=0.821 vs q0=0.806 (cap-binding)
- asymmetric (cost 0.5 vs 2.0): measured exposure shares = 0.75 vs 0.25 (low-cost provider takes the larger share)

## E4 selection effect (kappa)
- camera/simple r=0.3: kappa=1.4286 (survival retrievable 1.0 vs unretrievable 0.6907)
- camera/simple r=0.5: kappa=1.3333 (survival retrievable 0.6667 vs unretrievable 0.4948)
- camera/cpa r=0.3: kappa=1.0615 (survival retrievable 0.6875 vs unretrievable 0.7342)
- camera/cpa r=0.5: kappa=1.0279 (survival retrievable 0.5089 vs unretrievable 0.481)
- ev/cpa r=0.3: kappa=0.931 (survival retrievable 0.6432 vs unretrievable 0.8481)
- ev/cpa r=0.5: kappa=0.8689 (survival retrievable 0.4272 vs unretrievable 0.6962)
- shoes/simple r=0.3: kappa=1.4286 (survival retrievable 1.0 vs unretrievable 0.6869)
- shoes/simple r=0.5: kappa=2.0 (survival retrievable 1.0 vs unretrievable 0.4848)
- shoes/cpa r=0.3: kappa=1.0179 (survival retrievable 0.6912 vs unretrievable 0.7191)
- shoes/cpa r=0.5: kappa=1.004 (survival retrievable 0.4608 vs unretrievable 0.5955)

## R4 generator anchor (from logged rerun)
- spearman_exposure_vs_deltaASR_new: 0.939
- monotonicity_violations_pairs: 19
