"""Set up an NN architecture, run its training and test on the diode clipper data."""
import os
import soundfile as sf
from CoreAudioML.networks import SimpleRNN
from CoreAudioML.training import ESRLoss
from CoreAudioML.dataset import DataSet
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def main():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
    dataset = DataSet(data_dir='data')
    
    for subset_name, frame_len in zip(['train', 'validation', 'test'], [22050, 0, 0]):
        dataset.create_subset(subset_name, frame_len=frame_len)
        dataset.load_file(os.path.join(subset_name, 'diodeclip'), subset_name)

    EPOCH_COUNT = 20
    VALIDATE_EVERY = 5
    MINIBATCH_SIZE = 64
    SAMPLES_BETWEEN_UPDATES = 2048
    TEST_CHUNK = 100000
    BEST_MODEL_PATH = 'best_model.pth'
    RUNS_DIRECTORY = 'runs'
    
    network = SimpleRNN(input_size=1, output_size=1, unit_type="LSTM", hidden_size=8, skip=0, bias_fl=False)
    loss = ESRLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    logger = SummaryWriter(RUNS_DIRECTORY)
    
    best_validation_loss = 1
    
    for epoch in range(1, EPOCH_COUNT + 1):
        epoch_loss = network.train_epoch(dataset.subsets['train'].data['input'][0],
                                        dataset.subsets['train'].data['target'][0], loss, optimizer, bs=MINIBATCH_SIZE, up_fr=SAMPLES_BETWEEN_UPDATES)
                                        
        print(f"Epoch {epoch}/{EPOCH_COUNT} training loss: {epoch_loss}")
        logger.add_scalar('Loss/train', epoch_loss, epoch)
        
        if epoch % VALIDATE_EVERY == 0:
            validation_output, validation_loss = network.process_data(dataset.subsets['validation'].data['input'][0],
                                        dataset.subsets['validation'].data['target'][0], loss, TEST_CHUNK)

            print(f"Validation loss: {validation_loss}")
            logger.add_scalar('Loss/validation', validation_loss, epoch)
            
            if validation_loss < best_validation_loss:
                network_state = {
                    'model_state_dict': network.state_dict()
                }
                torch.save(network_state, BEST_MODEL_PATH)
                
            
    best_validation_model_state = torch.load(BEST_MODEL_PATH)
    network.load_state_dict(best_validation_model_state['model_state_dict'])

    test_output, test_loss = network.process_data(dataset.subsets['test'].data['input'][0],
                                                  dataset.subsets['test'].data['target'][0], loss, TEST_CHUNK)
    
    print(f"Test loss: {test_loss}")
    logger.add_scalar('Loss/test', test_loss, EPOCH_COUNT)

    sf.write("test_output.wav", test_output.cpu().numpy().flatten(), dataset.subsets['test'].fs)


if __name__ == '__main__':
    main()
