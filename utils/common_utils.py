import numpy as np
from config.config import Config
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _to_numpy(x):
	"""
	Convert input to a 1D numpy array. Handles numpy, pandas, and torch tensors.
	"""
	try:
		import torch
		torch_available = True
	except Exception:
		torch_available = False

	if x is None:
		return None
	if torch_available and isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy().ravel()
	try:
		import pandas as pd
		if isinstance(x, (pd.Series, pd.DataFrame)):
			return x.values.ravel()
	except Exception:
		# pandas not available or not a pandas object
		pass
	x = np.asarray(x)
	return x.ravel()

def compute_regression_metrics(y_true, y_pred, sample_weight=None, epsilon=1e-8):
	"""
	Compute common regression metrics between y_true and y_pred.

	Returns a dict:
	{
		'mse': ...,
		'rmse': ...,
		'mae': ...,
		'r2': ...,
		'mape': ...  # in percent
	}

	Inputs can be numpy arrays, pandas Series, or torch tensors.
	"""
	y_true_np = _to_numpy(y_true)
	y_pred_np = _to_numpy(y_pred)

	if y_true_np is None or y_pred_np is None:
		raise ValueError("y_true and y_pred must not be None")

	if y_true_np.shape != y_pred_np.shape:
		raise ValueError(f"Shapes of y_true {y_true_np.shape} and y_pred {y_pred_np.shape} do not match")

	# MSE
	mse = mean_squared_error(y_true_np, y_pred_np, sample_weight=sample_weight)
	rmse = float(np.sqrt(mse))
	# MAE
	mae = mean_absolute_error(y_true_np, y_pred_np, sample_weight=sample_weight)
	# R2
	r2 = r2_score(y_true_np, y_pred_np, sample_weight=sample_weight)
	# MAPE (in percent) - guard division by zero
	denom = np.maximum(np.abs(y_true_np), epsilon)
	mape = float(np.mean(np.abs((y_true_np - y_pred_np) / denom))) * 100.0

	return {
		"mse": float(mse),
		"rmse": rmse,
		"mae": float(mae),
		"r2": float(r2),
		"mape": mape
	}


def save_model_with_rotation(model, base_name='alexnet_bone_age_model', max_models=5):
	"""
	Saves the current model using a versioned filename and ensures
	only `max_models` files with the given base_name remain in model_files dir.
	"""
	model_save_dir = os.path.join(Config.BASEDIR, 'model_files')
	os.makedirs(model_save_dir, exist_ok=True)

	# find existing files matching base_name and allowed extensions
	allowed_exts = ('.keras', '.h5', 'joblib')
	existing = [os.path.join(model_save_dir, f) for f in os.listdir(model_save_dir)
				if f.startswith(base_name) and f.endswith(allowed_exts)]
	existing_sorted = sorted(existing, key=os.path.getmtime)  # oldest first

	# if there are already >= max_models, delete oldest so that after saving we'll have <= max_models
	if len(existing_sorted) >= max_models:
		num_to_delete = len(existing_sorted) - (max_models - 1)
		for i in range(num_to_delete):
			try:
				os.remove(existing_sorted[i])
			except Exception:
				# best-effort removal; ignore failures
				pass

	# create a unique versioned filename
	version = 1
	model_filename = f'{base_name}_v{version}.keras'
	model_save_path = os.path.join(model_save_dir, model_filename)
	while os.path.exists(model_save_path):
		version += 1
		model_filename = f'{base_name}_v{version}.keras'
		model_save_path = os.path.join(model_save_dir, model_filename)

	# save and return path
	model.save(model_save_path)
	print(f"Model saved successfully to: {model_save_path}")
	return model_save_path