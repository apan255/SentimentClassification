# SentimentClassification

Motive:
The dataset that we have was gleaned from their web site at
http://www.aueb.gr/users/ion/data/enron-spam/. Corpus has two folder one with spam emails
“Spam Folder” and no spam emails “Ham Folder”. The motive of classification is to detect Spam
emails from the Enron public email corpus.

Dataset Used:
The "raw" subdirectory contains the messages in their original form. Spam messages in non-Latin
encodings, ham messages sent by the owners of the mailboxes to themselves (sender in "To:",
"Cc:", or "Bcc" field), and a handful of virus-infected messages have been removed, but no other
modification has been made. The messages in the "raw" subdirectory are more than the
corresponding messages in the "preprocessed" subdirectory, because: (a) duplicates are preserved
in the "raw" form, and (b) during the preprocessing, ham and/or spam messages were randomly
subsampled to obtain the desired ham : spam ratios. See the paper for further details.
http://www.aueb.gr/users/ion/docs/ceas2006_paper.pdf
