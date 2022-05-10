class Hyperparams:
    
    
    prepro = False  

    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence

    # data
    data = "D:/finalProj/mainProj/hinglish/"
    # data = "/data/private/voice/nick"
    test_data = 'custom_sentences.txt'
    max_duration = 10.0

    # signal processing
    sr = 24000 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/01"
    logdir2="D:/finalProj/mainProj/logdir/01"
    sampledir = 'samples'
    batch_size = 50




