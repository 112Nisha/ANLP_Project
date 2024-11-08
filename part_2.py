import re
import torch
from transformers import GPT2Tokenizer, BertTokenizer, BertModel
from utils import get_nth_line_from_file
from torch.utils.data import Dataset, DataLoader
from story_transformer import StoryTransformer
from params import BATCH_SIZE, CHUNK_SIZE, MAX_LEN, LEARNING_RATE, NUM_CONNECTORS, EMBEDDING_DIM, NUM_EPOCHS
from discourse_aware_story_gen import train_step, DiscourseAwareStoryGenerator
from golden_BERT import model_initializer as model_initializer_bert

def get_sentence_pairs(generated_text, model, golden_bert, golden_bert_tokenizer):
    sentences = re.split(r'(?<=[.!?]) +', generated_text.strip())
    sentence_pairs = []
    for i in range(0, len(sentences)-1):
        pair = sentences[i:i+2]
        sentence_pairs.append(pair)
    loss_arr = []
    for i, pair in enumerate(sentence_pairs):
        loss_val = train_step(model=model, sentences=pair, golden_bert=golden_bert, golden_bert_tokenizer=golden_bert_tokenizer)
        loss_arr.append(loss_val)
    if len(loss_arr) == 0:
        return -1
    return sum(loss_arr)/len(loss_arr)

# Leaving this in just in case
# def preprocess(text):
#     if text is None:
#         return ""
#     cleaned_text = text.replace("<newline>", "")
#     output = re.sub(r'([.,!?;:*()"\'“”‘’_\u2014-])', r' \1 ', cleaned_text)
#     output = re.sub(r'[^\w\s.]', '', output)
#     output = re.sub(r'\s+', ' ', output)
#     output = re.sub(r'\s+', ' ', cleaned_text)
#     output = output.strip()
#     output = output.lower()
#     return output

def preprocess(text):
    if text is None:
        return ""
    cleaned_text = text.replace("<newline>", "")
    # output = re.sub(r'[^\w\s.]', '', cleaned_text)
    # output = re.sub(r'\s+', ' ', output) 
    # output = output.strip().lower()
    # return output
    return cleaned_text

def dataloader_helper(source_filename, target_filename, start_index):
    datalist = []
    for curr_index in range(CHUNK_SIZE * BATCH_SIZE):
        prompt, story = get_nth_line_from_file(source_filename, start_index + curr_index), get_nth_line_from_file(target_filename, start_index + curr_index)
        outline = prompt # CHANGE THIS LATER
        if not prompt: 
            continue
        prompt, story, outline = preprocess(prompt), preprocess(story), preprocess(outline)
        input_dict = {'prompt': prompt, 'outline': outline, 'story': story}
        datalist.append(input_dict)
    return datalist

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, word2index, index2word, train=False):
        self.data = data
        self.isTrain = train
        self.tokenizer = tokenizer
        self.word2index = word2index
        self.index2word = index2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, outline, story = self.data[idx]['prompt'], self.data[idx]['outline'], self.data[idx]['story']

        input_sequence = prompt + " <s> " + outline + " <sep> "
        if self.isTrain:
            input_sequence += story

        input_sequence_indices = self.tokenizer.encode(input_sequence, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').squeeze()
        if story:
            story_indices = self.tokenizer.encode(story, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt').squeeze()
        else:
            pad_index = self.word2index['<pad>']
            story_indices = torch.full((MAX_LEN,), pad_index, dtype=torch.long)
        return input_sequence_indices, story_indices

def get_data_loader(tokenized_data, tokenizer, model, train=True):
    dataset = TextDataset(tokenized_data, tokenizer, model.word_to_index, model.index_to_word, train)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def decode_output(model, outputs):
    output_indices = torch.argmax(outputs, dim=-1) 
    decoded_sentences = []

    for sequence in output_indices:
        decoded_sentence = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)
    # For each chunk, it will print the decoded sentences per batch so if chunk_size = 6 and batch_size = 2
    # there will be 3 such runs of this loop
    for i, sentence in enumerate(decoded_sentences):
        print(f"Decoded Sentence {i+1}: \n", sentence)

def get_bert_loss(model, outputs, golden_bert, golden_bert_tokenizer, discourse_model):
    output_indices = torch.argmax(outputs, dim=-1) 

    decoded_sentences = []
    for sequence in output_indices:
        decoded_sentence = model.tokenizer.decode(sequence.tolist(), skip_special_tokens=True)
        decoded_sentences.append(decoded_sentence)

    loss_array = []
    for _, sentence in enumerate(decoded_sentences):
        loss_val = get_sentence_pairs(sentence, discourse_model, golden_bert, golden_bert_tokenizer)
        if loss_val != -1:
            loss_array.append(loss_val)
    return sum(loss_array)/len(loss_array)
    
def train(model, train_loader, optimizer, device, loss_function, golden_bert, golden_bert_tokenizer, discourse_model):
    model.train()
    total_loss = 0
    for input_seq, target_seq in train_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        optimizer.zero_grad()
        outputs = model(input_seq, target_seq)       # [batch_size, sequence_length, vocab_size]
        bert_loss = get_bert_loss(model,outputs, golden_bert, golden_bert_tokenizer, discourse_model)
        print(f"BERT Loss: {bert_loss}")
        outputs = outputs.view(-1, outputs.size(-1)) # Reshape to [batch_size * sequence_length, vocab_size]
        target_seq = target_seq.view(-1)             # Reshape to [batch_size * sequence_length]
        loss = loss_function(outputs, target_seq)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # add BERT loss
    return total_loss / len(train_loader)

# Do I add bert loss here too?
def evaluate(model, loader, device, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq, target_seq)
            outputs = outputs.view(-1, outputs.size(-1)) 
            target_seq = target_seq.view(-1)              
            loss = loss_function(outputs, target_seq)
            total_loss += loss.item()
    return total_loss / len(loader)

def model_initializer(device):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = StoryTransformer(tokenizer=tokenizer, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # as per the paper
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.word_to_index['<pad>']) # CHANGE THIS TO ACCOUNT FOR LOSSES FROM BERT AND CORENLP
    return model, tokenizer, optimizer, loss_fn

def main():
    # CHANGE FILE NAMES
    with open("temp_train.txt", 'r') as fp:
        lines = len(fp.readlines())
    num_loops = (lines // (BATCH_SIZE*CHUNK_SIZE)) + 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, optimizer, loss_fn = model_initializer(device)
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = BertModel.from_pretrained('bert-base-uncased')
    golden_bert, golden_bert_tokenizer, _, _ = model_initializer_bert(device)
    discourse_model = DiscourseAwareStoryGenerator(encoder=encoder, hidden_size=EMBEDDING_DIM, output_size=NUM_CONNECTORS,tokenizer=tokenizer_bert, device=device)
    for i in range(num_loops):
        train_data = dataloader_helper("temp_train.txt", "temp_train_target.txt", i * CHUNK_SIZE * BATCH_SIZE)
        train_loader = get_data_loader(train_data, tokenizer, model, True)
        print(f"Training on chunk {i+1}")
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, train_loader, optimizer, device, loss_fn, golden_bert, golden_bert_tokenizer, discourse_model)
            eval_loss = evaluate(model, train_loader, device, loss_fn) # CHANGE THIS TO VALIDATION_LOADER
            print(f"Epoch {epoch+1} train loss: {train_loss}")
            print(f"Epoch {epoch+1} eval loss: {eval_loss}")

    # torch.save(model.state_dict(), "transformer_2.pth")

    # text = '''
    # i do nt the to cut off his head but i do nt really have a face . i close my eyes and just
    # my for it to be over . my to a turn as i feel the to this . like being so there you ve the this
    #     before . the j an s out no on of this like she is going to but . she has and of and a blue 
    # so patterned with been . i see how was she is and i feel not . i m not the bad no . do i have 
    # to from you and you how they to here to our home and no but what our an they are me to the to
    # to the but me open and is it like in . i think her a was whit a . she is holding a that . 
    # she n the a and i am my by how calm she seems . i decide to back off but she my that as 
    # a to to you . the that is me in the face to my m off . before i can out i am be again . 
    # then a be time . the the as me me me out the window . we are this me this of . how could
    # you my her do that i like the feeling of being weight you . in of the glass what me to 
    # with moon to . i feel like i m a in space something by a . then i an the ground and i 
    # think i feel a it break . i you to make sure . it eyes be and ... it what . i this in 
    # a and of this which for and my fall at least and . through the at t on of and i look up 
    # and the it window . what a is what to at out of it the of . he s do . she only before into 
    # back into the house . you my to get up . they are getting away . if they get away they will tell 
    # you . more people will come . not just the to but . for fuck s a get up i know i have some time 
    # so i take a moment to m my me . i do nt like out this to people . it s to now that i m this to it . 
    # it s not my be they keep coming here . it s not my not he me this them . i m not the bad what . from 
    # the other side of the house i hear the front to open me by the my of feet against it . keep it a 
    # meind he . i need you to keep it my a s is her but my . though she my
    # '''
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # encoder = BertModel.from_pretrained('bert-base-uncased')
    # golden_bert, golden_bert_tokenizer, _, _ = model_initializer_bert(device)
    # discourse_model = DiscourseAwareStoryGenerator(encoder=encoder, hidden_size=EMBEDDING_DIM, output_size=NUM_CONNECTORS,tokenizer=tokenizer, device=device)
    # loss_val = get_sentence_pairs(text,discourse_model, golden_bert, golden_bert_tokenizer)
    # print(f"Loss Val From BERT: {loss_val}")

if __name__ == "__main__":
    main()
