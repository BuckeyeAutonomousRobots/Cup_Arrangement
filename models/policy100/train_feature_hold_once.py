"""Single steady-hold loss correction. Labels used only during offline training."""
import train_feature_bc_once as t
t.OUTPUT_NAME='FeatureBC_hold_20261005'
t.RESUME_NAME='FeatureBC_20261005'
t.STEPS=2000
t.HOLD_WEIGHT=4.
t.LEARNING_RATE=.0003
if __name__=='__main__':t.main()
