# train / test data is in hdf5 format, we will convert it to numpy arrays and normalize
f_train = h5py.File('/content/drive/My Drive/cells/Chong_train_set.hdf5','r')
x_train, y_train = get_data(f_train)
x_train = normalize(x_train)

f_test = h5py.File('/content/drive/My Drive/cells/Chong_test_set.hdf5','r')
x_test, y_test = get_data(f_test)
x_test = normalize(x_test)
