from src import train, evaluation

# Train model
history, model = train.train_model()

# Evaluate performance
evaluation.plot_training_history(history)
