    # Try loading existing best model
    if os.path.exists('best_model.pt'):
        print("Loading existing best model...")
        try:
            checkpoint = torch.load('best_model.pt', map_location=device)
            network.load_state_dict(checkpoint)
            best_network.load_state_dict(checkpoint)
        except:
            print("Couldn't load existing model, initializing new one")
            network.apply(network.init_weights)
            best_network.load_state_dict(network.state_dict())
    else:
        print("No existing model found, initializing new one")
        network.apply(network.init_weights)
        best_network.load_state_dict(network.state_dict())
    