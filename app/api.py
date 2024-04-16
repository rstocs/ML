# Get model cutoff prob
_, cutoff_prob = predict(model_all, imputed_scaled_encoded_all, 0.75, True)
return cutoff_prob