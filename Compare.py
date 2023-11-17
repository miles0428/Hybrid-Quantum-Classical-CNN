import torch
import os

os.makedirs('figure', exist_ok=True)

# Load the results
hybrid_qnn_results = torch.load('data/HybridQNN/results.pt')
cnn_results = torch.load('data/CNN/results.pt')

# Plot the results
import matplotlib.pyplot as plt
plt.plot(hybrid_qnn_results['train_loss'], label='QCNN-Trained')
plt.plot(cnn_results['train_loss'], label='CNN-Trained')
plt.plot(hybrid_qnn_results['test_loss'], label='QCNN-Test')
plt.plot(cnn_results['test_loss'], label='CNN-Test')
plt.title('Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('figure/loss.png')
plt.clf()

plt.plot(hybrid_qnn_results['train_accu'], label='QCNN-Trained')
plt.plot(cnn_results['train_accu'], label='CNN-Trained')
plt.plot(hybrid_qnn_results['test_accu'], label='QCNN-Test')
plt.plot(cnn_results['test_accu'], label='CNN-Test')
plt.title('Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('figure/accuracy.png')
plt.clf()

print('Original')
print('QCNN')
print(hybrid_qnn_results)
print('CNN')
print(cnn_results)

hybrid_qnn_results = torch.load('data/HybridQNN_T/results.pt')
cnn_results = torch.load('data/CNN_T/results.pt')

# Plot the results
import matplotlib.pyplot as plt
plt.plot(hybrid_qnn_results['train_loss'], label='QCNN-Trained')
plt.plot(cnn_results['train_loss'], label='CNN-Trained')
plt.plot(hybrid_qnn_results['test_loss'], label='QCNN-Test')
plt.plot(cnn_results['test_loss'], label='CNN-Test')
plt.title('Loss-Transfer')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('figure/loss_T.png')
plt.clf()

plt.plot(hybrid_qnn_results['train_accu'], label='QCNN-Trained')
plt.plot(cnn_results['train_accu'], label='CNN-Trained')
plt.plot(hybrid_qnn_results['test_accu'], label='QCNN-Test')
plt.plot(cnn_results['test_accu'], label='CNN-Test')

print('Transfer')
print('QCNN')
print(hybrid_qnn_results)
print('CNN')
print(cnn_results)

plt.title('Accuracy-Transfer')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('figure/accuracy_T.png')
plt.clf()
# 
# import torch
# 
# model1 = torch.load('data/HybridQNN_T/model.pt')
# model2 = torch.load('data/HybridQNN/model.pt



