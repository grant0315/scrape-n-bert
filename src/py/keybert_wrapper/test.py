import pickle



filename = "p_rep_docs"
outfile = open(filename, 'wb')
pickle.dump(rep_docs, outfile)
outfile.close()