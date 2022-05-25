from summarizer import Summarizer,TransformerSummarizer

article = """The problem is affecting people using the older
versions of the PlayStation 3, called the "Fat"
model.The problem isn't affecting the newer PS3
Slim systems that have been on sale since
September last year.Sony have also said they are
aiming to have the problem fixed shortly but is
advising some users to avoid using their console
for the time being."We hope to resolve this
problem within the next 24 hours," a statement
reads. "In the meantime, if you have a model other
than the new slim PS3, we advise that you do not
use your PS3 system, as doing so may result in
errors in some functionality, such as recording
obtained trophies, and not being able to restore
certain data."We believe we have identified that
this problem is being caused by a bug in the clock
functionality incorporated in the system."The
PlayStation Network is used by millions of people
around the world.It allows users to play their
friends at games like Fifa over the internet and
also do things like download software or visit
online stores."""

bert_model = Summarizer()
bert_summary = ''.join(bert_model(article, min_length=60))
print(bert_summary)