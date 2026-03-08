import numpy as np
import matplotlib.pyplot as plt
import random
import struct
from sklearn.manifold import TSNE

#Task 1
def load_idx_file(file_path):
    with open(file_path, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape).tolist()
        return data

def get_train_test_data(image_file, label_file,amount):
    train_images = load_idx_file(image_file)
    train_labels = load_idx_file(label_file)

    subset_image = []
    subset_label = []

    zero_count = 0
    one_count  = 0
    two_count  = 0
    start = random.randint(0, int(len(train_labels)/2))

    for i in range(start,len(train_labels)):
        if train_labels[i] == 1 and one_count < amount:
            subset_image.append(train_images[i])
            subset_label.append(train_labels[i])
            one_count += 1
        if train_labels[i] == 2 and two_count < amount:
            subset_image.append(train_images[i])
            subset_label.append(train_labels[i])
            two_count += 1
        if train_labels[i] == 0 and zero_count < amount:
            subset_image.append(train_images[i])
            subset_label.append(train_labels[i])
            zero_count += 1

        if zero_count == amount and one_count == amount and two_count == amount:
            break

    return subset_image, subset_label

def stack_colums_in_data(dataset):
    for i in range(len(dataset)):
        stacked_image = []
        orignal_image = dataset[i]

        for j in range(len(orignal_image[0])):
            for k in range(len(orignal_image)):
                stacked_image.append(orignal_image[k][j]/255)

        dataset[i] = stacked_image

    return dataset

#Task 2
def compute_mean(data):
    data_list = data

    n_samples = len(data_list)
    n_features = len(data_list[0])
    
    mean_vector = [0.0] * n_features
    for row_idx in range(n_samples):
        row = data_list[row_idx]
        for col_idx in range(n_features):
            mean_vector[col_idx] += row[col_idx]
    
    for i in range(n_features):
        mean_vector[i] = mean_vector[i] / n_samples
        
    return mean_vector

def compute_mle(images, label):
    parameters = {}
    classes = [0, 1, 2]

    for c in classes:
        images_c = []
        for i in range(len(label)):
            if(label[i] == c):
                images_c.append(images[i])
        N_c = len(images_c) - 1
        
        mu_list = compute_mean(images_c)

        images_c = np.array(images_c)        
        mu = np.array(mu_list)

        images_centered = images_c - mu
        sigma = (images_centered.T @ images_centered) / N_c
        
        # Added small noise for invertibility
        sigma = sigma + np.eye(len(mu)) * 1e-5

        parameters[c] = {
            'mean': mu,
            'cov': sigma,
            'prior':0.33
        }
    return parameters

#Task 3
def calculate_lda(x, avg_cov_I, mu, mu_T, prior):
    return avg_cov_I @ mu_T @ x - (0.5)*(mu_T @ avg_cov_I @ mu) + np.log(prior)

def calculate_qda(x, cov_I, mu,  prior, minus_half_ln_mod_cov):
    return minus_half_ln_mod_cov - (0.5)*((x - mu).T @ cov_I @ (x-mu)) + np.log(prior)

def predict_lda(X_test, params):
    predictions = []
    
    shared_cov = sum(p['cov'] for p in params.values()) / len(params)  
    avg_cov_I = np.linalg.pinv(shared_cov)
    
    for x in X_test:
        scores = {}
        for c in params:
            mu = params[c]['mean']
            prior = params[c]['prior']
            mu_T = mu.T
            scores[c] = calculate_lda(x, avg_cov_I, mu, mu_T, prior)
            
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

def predict_qda(X_test, params):
    predictions = []
    
    class_utils = {}
    for c in params:
        cov = params[c]['cov']
        
        minus_half_ln_mod_cov = -0.5 * np.linalg.slogdet(cov)[1]
        
        class_utils[c] = {
            'cov_I': np.linalg.pinv(cov),
            'minus_half_ln_mod_cov': minus_half_ln_mod_cov,
            'mean': params[c]['mean'],
            'prior': params[c]['prior']
        }
    
    for x in X_test:
        scores = {}
        for c in params:
            utils = class_utils[c]
            
            scores[c] = calculate_qda(x, utils['cov_I'], utils['mean'], utils['prior'], utils['minus_half_ln_mod_cov'])
            
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

def calculate_accuracy(prediction, true_value):
    entries = 0
    correct = 0
    for i in range(len(prediction)):
        if (prediction[i] == true_value[i]):
            correct += 1
        entries += 1
    return correct/entries

def plot_tsne(X_train, y_train, X_test, y_test):
    print("\n   Generating t-SNE plots")
    X_combined = np.vstack([X_train, X_test])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_combined)

    X_train_emb = X_embedded[:len(y_train)]
    X_test_emb = X_embedded[len(y_train):]

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_train_emb[:, 0], X_train_emb[:, 1], c=y_train, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=['0', '1', '2'])
    plt.title("t-SNE: Train Set")
    
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_test_emb[:, 0], X_test_emb[:, 1], c=y_test, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter2.legend_elements()[0], labels=['0', '1', '2'])
    plt.title("t-SNE: Test Set")
    
    plt.tight_layout()
    plt.show()

def print_sample_discriminant(test_x, test_y, model_params, sample_idx):
    print(f"\n   Discriminant Values for Test Sample {sample_idx}")
    
    sample_x = test_x[sample_idx]
    true_label = test_y[sample_idx]
    
    print(f"     True Label: {true_label}")
    
    print("\n   QDA Discriminant Scores:")
    for c in [0, 1, 2]:
        params = model_params[c]
        cov = params['cov']
        mu = params['mean']
        prior = params['prior']
        
        cov_I = np.linalg.pinv(cov)
        minus_half_ln_mod_cov = -0.5 * np.linalg.slogdet(cov)[1]
        
        score = calculate_qda(sample_x, cov_I, mu, prior, minus_half_ln_mod_cov)
        
        print(f"     Class {c} Score: {score:.4f}")
        
    print("\n   LDA Discriminant Scores:")
    
    shared_cov = sum(p['cov'] for p in model_params.values()) / len(model_params)
    avg_cov_I = np.linalg.pinv(shared_cov)
    
    for c in [0, 1, 2]:
        mu = model_params[c]['mean']
        prior = model_params[c]['prior']
        mu_T = mu.T        
        score = calculate_lda(sample_x, avg_cov_I, mu, mu_T, prior)
        print(f"     Class {c} Score: {score:.4f}")
 


# Assignment 2 specific functions
def get_pca(dataset, target_var):
    images_c = dataset

    N_c = len(images_c) - 1
    mu_list = compute_mean(images_c)

    images_c = np.array(images_c)        
    mu = np.array(mu_list)

    images_centered = images_c - mu
    sigma = (images_centered.T @ images_centered) / N_c

    eigenvalues, eigenvectors = np.linalg.eig(sigma)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    total_var   = np.sum(eigenvalues)
    current_var = 0
    idx = -1

    for i in range(len(eigenvalues)):
        current_var += eigenvalues[i]
        if (current_var >= target_var*total_var):
            idx = i
            break

    return eigenvectors[:, :idx+1]

def get_fda(images, label):
    classes = [0, 1, 2]
    
    global_mean = np.array(compute_mean(images))
    features = train_x.shape[1]
    S_w = np.zeros((features, features))
    S_b = np.zeros((features, features))

    for c in classes:
        images_c = []
        for i in range(len(label)):
            if(label[i] == c):
                images_c.append(images[i])
        N_c = len(images_c)

        mu = np.array(compute_mean(images_c))
        images_c = np.array(images_c)

        mean_centered = mu - global_mean
        
        S_b += (mean_centered.T @ mean_centered) * N_c

        for x in images_c:
            x_diff = (x - mu)
            S_w += x_diff.T @ x_diff

    # Added small noise for inverse
    S_w = S_w + np.eye(len(mu)) * 1e-5
    S_w_inv = np.linalg.inv(S_w)

    eigenvalues, eigenvectors = np.linalg.eig(S_w_inv @ S_b)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors[:, :2]

def reconstruct_pca(X_pca, W_pca, mu):
    return (X_pca @ W_pca.T) + mu

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def plot_reconstructed_samples(original, reconstructed, num_samples=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(original[i].reshape((28, 28), order='F'), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(reconstructed[i].reshape((28, 28), order='F'), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.suptitle(f"PCA Reconstruction (Showing {num_samples} Samples)")
    plt.tight_layout()
    plt.show()

def plot_2d_scatter(X_2d, y, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    markers = ['o', 's', '^']
    for c, col, mk in zip([0, 1, 2], colors, markers):
        plt.scatter(X_2d[y == c, 0], X_2d[y == c, 1], 
                    c=col, marker=mk, label=f'Class {c}', alpha=0.7, edgecolors='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    
    print("\n================= Evaluate and Compare Performance =================")
    
    # Load Data 
    train_x, train_y = get_train_test_data('train-images.idx3-ubyte','train-labels.idx1-ubyte', 100)
    test_x, test_y = get_train_test_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 100)

    train_x = np.array(stack_colums_in_data(train_x))
    test_x = np.array(stack_colums_in_data(test_x))        
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    mu_train = np.array(compute_mean(train_x))

    # --- 1. Apply FDA on the test set. Compute classification accuracy of LDA and QDA ---
    print("\n--- 1. FDA Pipeline ---")
    W_fda = get_fda(train_x, train_y)
    
    train_fda = train_x @ W_fda
    test_fda = test_x @ W_fda
    
    fda_params = compute_mle(train_fda, train_y)
    
    print("FDA - LDA Accuracy:")
    print(f"  Train Accuracy: {calculate_accuracy(predict_lda(train_fda, fda_params), train_y):.4f}")
    print(f"  Test Accuracy:  {calculate_accuracy(predict_lda(test_fda, fda_params), test_y):.4f}")
    
    print("\nFDA - QDA Accuracy:")
    print(f"  Train Accuracy: {calculate_accuracy(predict_qda(train_fda, fda_params), train_y):.4f}")
    print(f"  Test Accuracy:  {calculate_accuracy(predict_qda(test_fda, fda_params), test_y):.4f}")

    # Plot FDA 2D Space
    plot_2d_scatter(train_fda, train_y, "FDA: 2D Projection of Training Data", "Discriminant 1", "Discriminant 2")


    # --- 2. Apply PCA and then apply LDA (75% Variance) & Reconstruction ---
    print("\n--- 2. PCA Pipeline (75% Variance) ---")
    W_pca_75 = get_pca(train_x, 0.75)
    
    train_pca_75 = (train_x - mu_train) @ W_pca_75
    test_pca_75 = (test_x - mu_train) @ W_pca_75
    
    pca_params_75 = compute_mle(train_pca_75, train_y)
    
    print("PCA (75%) - LDA Accuracy:")
    print(f"  Train Accuracy: {calculate_accuracy(predict_lda(train_pca_75, pca_params_75), train_y):.4f}")
    print(f"  Test Accuracy:  {calculate_accuracy(predict_lda(test_pca_75, pca_params_75), test_y):.4f}")

    # 2a. Reconstruct and show the result for 5 samples
    print("\n--- PCA Reconstruction (75% Variance) ---")
    test_reconstructed_75 = reconstruct_pca(test_pca_75, W_pca_75, mu_train)
    mse_75 = calculate_mse(test_x, test_reconstructed_75)
    print(f"Mean Squared Error (MSE) of 75% Variance Reconstruction: {mse_75:.4f}")
    
    # Plot the first 5 test samples
    plot_reconstructed_samples(test_x[:5], test_reconstructed_75[:5], num_samples=5)


    # --- 3. Analyse the accuracy by changing the variance to 90% ---
    print("\n--- 3. PCA Pipeline (90% Variance) ---")
    W_pca_90 = get_pca(train_x, 0.90)
    
    train_pca_90 = (train_x - mu_train) @ W_pca_90
    test_pca_90 = (test_x - mu_train) @ W_pca_90
    
    pca_params_90 = compute_mle(train_pca_90, train_y)
    
    print("PCA (90%) - LDA Accuracy:")
    print(f"  Train Accuracy: {calculate_accuracy(predict_lda(train_pca_90, pca_params_90), train_y):.4f}")
    print(f"  Test Accuracy:  {calculate_accuracy(predict_lda(test_pca_90, pca_params_90), test_y):.4f}")

    print("\nPCA (90%) - QDA Accuracy:")
    print(f"  Train Accuracy: {calculate_accuracy(predict_qda(train_pca_90, pca_params_90), train_y):.4f}")
    print(f"  Test Accuracy:  {calculate_accuracy(predict_qda(test_pca_90, pca_params_90), test_y):.4f}")


    # --- 4. Analyse the accuracy by using only first two principal components ---
    print("\n--- 4. PCA Pipeline (First 2 Principal Components) ---")
    W_pca_2 = W_pca_90[:, :2]
    
    train_pca_2 = (train_x - mu_train) @ W_pca_2
    test_pca_2 = (test_x - mu_train) @ W_pca_2
    
    pca_params_2 = compute_mle(train_pca_2, train_y)
    
    print("PCA (2 Components) - LDA Accuracy:")
    print(f"  Train Accuracy: {calculate_accuracy(predict_lda(train_pca_2, pca_params_2), train_y):.4f}")
    print(f"  Test Accuracy:  {calculate_accuracy(predict_lda(test_pca_2, pca_params_2), test_y):.4f}")

    print("\nPCA (2 Components) - QDA Accuracy:")
    print(f"  Train Accuracy: {calculate_accuracy(predict_qda(train_pca_2, pca_params_2), train_y):.4f}")
    print(f"  Test Accuracy:  {calculate_accuracy(predict_qda(test_pca_2, pca_params_2), test_y):.4f}")

    # Plot PCA 2D Space
    plot_2d_scatter(train_pca_2, train_y, "PCA: 2D Projection (First 2 Components)", "Principal Component 1", "Principal Component 2")
