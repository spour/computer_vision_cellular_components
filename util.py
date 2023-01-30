# returns data and one-hot labels in np.array format
# data is in format Number-Height-Width-Channel (NHWC)
# takes hdf5 file as input
def get_data(f):
  x, y = f['data1'], f['Index1']
  x = np.array(x).reshape(-1, 2, 64, 64)
  x = np.transpose(x, (0, 2, 3, 1))
  y = np.array(y)
  return x, y


# performs min-max normalization of the data
def normalize(x):
  x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
  return x_norm


# create a dictionary: class name -> label
# use original hdf5 file and np array of one-hot labels
def get_class_names_dict(f, y):
  y_names = np.array(f['label_names'])
  y_names = np.array([x.decode("utf-8").lower() for x in y_names])
  sparse_y = y.argmax(axis=1)
  label_names = set(y_names)
  names_dict = {}

  for name in label_names:
    idx = np.where(y_names==name)[0][0]
    names_dict[name] = sparse_y[idx]
  
  return names_dict


# plot examples of each class in the dataset 
# use data np array in NHWC format and dictionary (class name -> label)
def plot_class_examples(x, y, names_dict):
  label_names = list(names_dict)
  n_classes, n_examples = len(label_names), 5
  fig, axes = plt.subplots(nrows=n_classes, ncols=n_examples, figsize=(10, 2*len(label_names)))

  for class_name in list(names_dict):
    k = names_dict[class_name]
    axes[k, 0].set_ylabel("{} ({})".format(class_name, names_dict[class_name]))

    idx = np.where(y.argmax(axis=1)==k)[0]
    
    for j in range(n_examples):
      axes[k, j].imshow(x[idx[j],:,:,0])
      axes[k,j].set_xticks([])
      axes[k,j].set_yticks([])
  plt.show()


# get number of examples from each class
# use one-hot labels array 
def get_class_counts(y):
  unique, counts = np.unique(y.argmax(axis=1), return_counts=True)
  return dict(zip(unique, counts)) 



# plot barplot with number of samples per each label class
# this visualization should help to assess if dataset is balanced
def plot_class_counts(y):
  val_dict = get_class_counts(y)
  plt.bar(list(val_dict), list(val_dict.values()))
  plt.xticks(list(val_dict))
  plt.show()


# get a selection of samples representing each class
def get_selected_set(x, y):
  labels = sorted(set(y.argmax(axis=1)))
  selected_imgs = []

  for k in labels:
    idx = np.where(y.argmax(axis=1)==k)[0]
    selected_imgs.append(x[idx[0]:idx[0]+1])
  
  selected_imgs = np.vstack(selected_imgs)
  return selected_imgs
