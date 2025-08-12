Hi! Our application runs ML models. We store them in a particular folder structure. See ~/invokeai-4.0.0/models for an example. 

This isn't great - it means we cannot install models with the same file name (even when they are different), and the user-friendly folder structure tempts users into manipulating the files. Because model files are recorded in our database, if users go in and rename or move the files, we end up with db-fs sync issues.

Models may be single-file (e.g. .safetensors, .pth, or .ckpt files) or folders (e.g. diffusers-format models).

I'd like to explore migrating to a normalized structure, where every model is in its own folder. The folder name is the unique key of the model (PK in the db).

Please first review the architecture of the application, how we store models, and write up a proposal to migrate to this normalized structure.

Use @agent-python-pro, @agent-database-admin to formulate the proposal, writing it to PROPOSAL.md. Then do a self-review with @agent-architect-reviewer, writing the review to REVIEW.md.
