def sample():
    i = 0
    random_predictor = random.choice(ngrams)
    headline = ""
    for word in random_predictor:
        headline += " "+ind2word[word]
    
    while(len(random_predictor)<max_sequence_len):
        print(headline)
        random_padded = [torch.tensor(random_predictor), predictor]

        random_padded = pad_sequence(random_padded, batch_first=True)

        input_random = random_padded[0]
        input_random = input_random.unsqueeze(0)
        input_random = input_random.unsqueeze(-1)
        input_random = input_random.float()
        output =  model(input_random)

        topv, topi = output.topk(1)
        topi = topi[0][0]

        word = ind2word[topi.item()]
        headline += " "+word
        random_predictor.append(topi.item())
    return headline

    





