#__all__ = ["getVocabList"]
import io
import re
import numpy as np

def getVocabList():
    vocab = {}
    with io.open('vocab.txt', 'r') as f:
        contents_list = f.readlines();
        for content in contents_list:
            k, v = content.replace('\n', '').split('\t');
            vocab[k] = v;
    return vocab

# returns a linear kernel between x1 and x2 and returns the value in sim
def linearKernel(x1, x2):
    # Ensure that x1 and x2 are column vectors
    x1 = np.ravel(x1);
    x2 = np.ravel(x2);
    # Compute the kernel
    #return x1.T @ x2
    return x1.T @ x2

def svmTrain(X, Y, C, kernel='linear', sigma=2, tol=1e-3, max_passes=5):

    # SVMTRAIN Trains an SVM classifier using a simplified version of the SMO algorithm and returns trained model. 
    #
    # X is the matrix of training examples.  
    # Each row is a training example, and the jth column holds the jth feature.  
    # Y is a column matrix containing 1 for positive examples and 0 for negative examples.  
    # C is the standard SVM regularization parameter.  
    # tol is a tolerance value used for determining equality of floating point numbers. 
    # max_passes controls the number of iterations over the dataset (without changes to alpha) before the algorithm quits.
    #
    # Note: This is a simplified version of the SMO algorithm for training SVMs. 
    #       In practice, if you want to train an SVM classifier, we recommend using an optimized package such as:  
    #           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    #           SVMLight (http://svmlight.joachims.org/)
    #
   
    # Data parameters
    m, n = X.shape

    # Map 0 to -1
    Y = Y+(-1)+Y

    # Variables
    alphas = np.zeros([m, 1]);
    b = 0;
    E = np.zeros([m, 1]);
    passes = 0;
    eta = 0;
    L = 0;
    H = 0;

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets gracefully will _not_ do this)
    # 
    # We have implemented optimized vectorized version of the Kernels here 
    # so that the svm training will run faster.
    if kernel == 'linear':
        # Vectorized computation for the Linear Kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = X @ X.T;
    elif kernel == 'gaussian':
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X2 = np.array([np.sum(np.power(X, 2), 1)]).T
        K = X2.T - 2 * (X @ X.T)
        K = X2 + K
        K = np.power(gaussianKernel(1, 0, sigma), K)
    else:
        # Pre-compute the Kernel Matrix
        # The following can be slow due to the lack of vectorization
        K = np.zeros([m, m]);
        for i in range(m):
            for j in range(m):
                 K[i,j] = linearKernel(X[i,:].T, X[j,:].T);
                 #K[i,j] = gaussianKernel(X[i,:].T, X[j,:].T, sigma); 
                 K[j,i] = K[i,j]; # the matrix is symmetric

    # Train
    print('\nTraining ...');
    dots = 12;
    while passes < max_passes:

        num_changed_alphas = 0;
        for i in range(m):

            # Calculate Ei = f(x(i)) - y(i) using (2). 
            # E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
            E[i] = b + np.sum(np.multiply(np.multiply(alphas, Y), K[:,i:i+1])) - Y[i];
            if (Y[i]*E[i] < -tol and alphas[i] < C) or (Y[i]*E[i] > tol and alphas[i] > 0):

                # In practice, there are many heuristics one can use to select
                # the i and j. In this simplified code, we select them randomly.
                j = int(np.ceil((m-1) * np.random.rand()));
                while j == i:  # Make sure i \neq j
                    j = int(np.ceil((m-1) * np.random.rand()));

                # Calculate Ej = f(x(j)) - y(j) using (2).

                E[j] = b + np.sum(np.multiply(np.multiply(alphas, Y), K[:,j:j+1])) - Y[j];

                # Save old alphas
                alpha_i_old = alphas[i].copy();
                alpha_j_old = alphas[j].copy();

                # Compute L and H by (10) or (11). 
                if Y[i] == Y[j]:
                    var_list = alphas[j] + alphas[i] - C
                    L = (0>=var_list) * 0 + (0<var_list) * var_list # equal Octave's max()

                    var_list = alphas[j] + alphas[i]
                    H = (C<var_list) * C + (C>=var_list) * var_list # equal Octave's min()
                else:
                    var_list = alphas[j] - alphas[i]
                    L = (0>=var_list) * 0 + (0<var_list) * var_list # equal Octave's max()

                    var_list = alphas[j] - alphas[i] + C
                    H = (C<var_list) * C + (C>=var_list) * var_list # equal Octave's min()     

                if L == H:
                    # continue to next i. 
                    continue;

                # Compute eta by (14).
                eta = 2 * K[i,j] - K[i,i] - K[j,j];
                
                if eta >= 0:
                    #% continue to next i. 
                    continue;

                # Compute and clip new value for alpha j using (12) and (15).
                alphas[j] = alphas[j] - (Y[j] * (E[i] - E[j])) / eta;

                # Clip
                var_list = alphas[j]
                alphas[j] = (H<var_list) * H + (H>=var_list) * var_list # equal Octave's min()   

                var_list = alphas[j]
                alphas[j] = (L>=var_list) * L + (L<var_list) * var_list # equal Octave's max()
                
                # Check if change in alpha is significant
                if np.abs(alphas[j] - alpha_j_old) < tol:
                    # continue to next i. 
                    # replace anyway
                    alphas[j] = alpha_j_old.copy();
                    continue;
                # Determine value for alpha i using (16). 
                alphas[i] = alphas[i] + Y[i]*Y[j]*(alpha_j_old - alphas[j]);

                # Compute b1 and b2 using (17) and (18) respectively. 
                b1 = b - E[i] \
                     - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j].T \
                     - Y[j] * (alphas[j] - alpha_j_old) *  K[i,j].T;
                b2 = b - E[j] \
                     - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j].T \
                     - Y[j] * (alphas[j] - alpha_j_old) *  K[j,j].T;

                # Compute b by (19). 
                if 0 < alphas[i] and alphas[i] < C:
                    b = b1;
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2;
                else:
                    b = (b1+b2)/2;

                num_changed_alphas = num_changed_alphas + 1;

        if num_changed_alphas == 0:
            passes = passes + 1;
        else:
            passes = 0;

        print('.', end='');
        dots = dots + 1;
        if dots > 78:
            dots = 0;
            print('\n');
    print(' Done! \n\n');
    
    # Save the model
    model = {}
    idx = np.where(alphas>0)[0];
    model['X'] = X[idx,:]
    model['y'] = Y[idx];
    model['kernelFunction'] = kernel;
    model['sigma'] = sigma;
    model['b'] = b;
    model['alphas'] = alphas[idx];
    model['w'] = (np.multiply(alphas, Y).T @ X).T;
    
    return model

#  Evaluating the Gaussian Kernel
def gaussianKernel(x1, x2, sigma):
    #  returns a gaussian kernel between x1 and x2 and returns the value in sim
    #  Ensure that x1 and x2 are row vectors x1 = x1(:); x2 = x2(:);
    x1 = np.ravel(x1);
    x2 = np.ravel(x2);
    # return the similarity between x1 and x2 computed 
    # using a Gaussian kernel with bandwidth sigma
    return np.exp(-np.sum(np.power(x1-x2, 2))/2/sigma/sigma)

def svmPredict(model, X):
    # Returns a vector of predictions using a trained SVM model (svmTrain). 
    # X is a mxn matrix where there each example is a row. 
    # model is a svm model returned from svmTrain.
    # predictions pred is a m x 1 column of predictions of {0, 1} values.
    #

    # Check if we are getting a column vector, 
    # if so, then assume that we only need to do prediction for a single example
    if X.shape[1] == 1:
        X = X.T; # Examples should be in rows

    # Dataset 
    m = X.shape[0];
    p = np.zeros([m, 1]);
    pred = p.copy();

    if model['kernelFunction'] == 'linear':
        # We can use the weights and bias directly if working with the linear kernel
        p = X @ model['w'] + model['b'];
    elif model['kernelFunction'] == 'gaussian':
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X1 = np.array([np.sum(np.power(X, 2), axis=1)]).T;
        X2 = np.sum(np.power(model['X'], 2), axis=1).T;
        K = (-2) * (X @ model['X'].T) + X2 + X1;
        K = np.power(gaussianKernel(1, 0, model['sigma']), K);
        K = np.multiply(model['y'].T, K)
        K = np.multiply(model['alphas'].T, K);
        p = np.sum(K, axis=1);
    else:
        # Other Non-linear kernel  该部分代码未验证过
        for i in range(m):
            prediction = 0;
            for j in range(model['X'].shape[0]):
                prediction = prediction + model['alphas'][j] @ model['y'][j] @ gaussianKernel(X[i,:].T, model['X'][j,:].T, model['sigma']);
            p[i] = prediction + model['b'];

    # Convert predictions into 0 / 1
    pred[np.where(p>=0)[0]] = 1;
    pred[np.where(p< 0)[0]] = 0;

    return pred

def dataset3Params(X, y, Xval, yval):
    # Returns your choice of C and sigma for Part 3 of the exercise
    # where you select the optimal (C, sigma) learning parameters to use for SVM with RBF kernel.
    # You should complete this function to return the optimal C and sigma based on a cross-validation set.

    # You need to return the following variables correctly.
    C = 1;
    sigma = 0.3;

    # Instructions:
    # Fill in this function to return the optimal C and sigma learning parameters found using the cross validation set.
    # You can use svmPredict to predict the labels on the cross validation set. For example,
    # predictions = svmPredict(model, Xval);
    # will return the predictions on the cross validation set.
    # Note: You can compute the prediction error using mean(double(predictions ~= yval))
    wid = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]);
    errors = np.zeros([wid.shape[0], wid.shape[0]]);
    for i in range(wid.shape[0]):
        for j in range(wid.shape[0]):
            model = svmTrain(X, y, C=wid[i], kernel='gaussian', sigma=wid[j], tol=1e-3, max_passes=5)
            predictions = svmPredict(model, Xval);
            errors[i,j] = np.mean((predictions != yval));

    error = np.min(errors);
    i, j = np.where(errors == error);
    C = wid[i];
    sigma = wid[j];

    return C, sigma, errors

def readFile(filename):
    file_contents = ''
    with io.open(filename, 'r') as f:
        contents_list = f.readlines();
        for content in contents_list:
            file_contents += content
    return file_contents

def processEmail(email_contents):
    # preprocesses the body of an email and returns a list of indices of the words contained in the email. 

    # Load Vocabulary
    vocabList = getVocabList();
    

    # Init return value
    features = np.zeros([1, len(vocabList.keys())]);

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # email_contents = email_contents(hdrstart(1):end);
    
    # Lower case
    email_contents = email_contents.lower();
    
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub(pattern='<[^<>]+>', repl=' ', string=email_contents, count=0, flags=0)
    
    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(pattern='[0-9]+', repl='number', string=email_contents, count=0, flags=0)
    
    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(pattern='(http|https)://[^\s]*', repl='httpaddr', string=email_contents, count=0, flags=0)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(pattern='[^\s]+@[^\s]+', repl='emailaddr', string=email_contents, count=0, flags=0)

    # Handle $ sign
    email_contents = re.sub(pattern='[$]+', repl='dollar', string=email_contents, count=0, flags=0)

    # split string to list
    email_contents = re.split("[ @$/#.-:&*+=\[\]!(){},'\">_<;%?\t\n]", email_contents)
 
    # remove [^a-zA-Z0-9]
    for i in range(97, 123):
        while chr(i) in email_contents: email_contents.remove(chr(i))
    for i in range(65, 91):
        while chr(i) in email_contents: email_contents.remove(chr(i))
    for i in range(10):
        while chr(i) in email_contents: email_contents.remove(chr(i))
    while '' in email_contents: email_contents.remove('')

    # Look up the word in the dictionary and add to word_indices if
    # found
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to add the index of str to
    #               word_indices if it is in the vocabulary. At this point
    #               of the code, you have a stemmed word from the email in
    #               the variable str. You should look up str in the
    #               vocabulary list (vocabList). If a match exists, you
    #               should add the index of the word to the word_indices
    #               vector. Concretely, if str = 'action', then you should
    #               look up the vocabulary list to find where in vocabList
    #               'action' appears. For example, if vocabList{18} =
    #               'action', then, you should add 18 to the word_indices 
    #               vector (e.g., word_indices = [word_indices ; 18]; ).
    # 
    # Note: vocabList{idx} returns a the word with index idx in the
    #       vocabulary list.
    # 
    # Note: You can use strcmp(str1, str2) to compare two strings (str1 and
    #       str2). It will return 1 only if the two strings are equivalent.
    #
        
    for i in range(len(email_contents)):
        for j in range(len(vocabList.keys())):
            if email_contents[i] in vocabList[str(j+1)]:
                features[0, j] = 1;
    
    return features


