# Custom Node Manager

The Custom Node Manager allows you to install, manage, and remove community node packs directly from the InvokeAI UI — no manual file copying required.

## Accessing the Node Manager

Click the **Nodes** tab (circuit icon) in the left sidebar, between Models and Queue.

## Installing a Node Pack

1. Navigate to the **Nodes** tab
2. On the right panel, select the **Git Repository URL** tab
3. Paste the Git URL of the node pack (e.g. `https://github.com/user/my-node-pack.git`)
4. Click **Install**

The installer will:

- Clone the repository into your `nodes` directory
- Load the nodes immediately — no restart needed
- Import any workflow `.json` files found in the repository into your workflow library (tagged with `node-pack:<name>` for easy filtering)

The install progress and results are shown in the **Install Log** at the bottom of the panel.

### Installing Python Dependencies

The installer does **not** automatically run `pip install` for `requirements.txt` or `pyproject.toml`. Auto-installing dependencies into the running InvokeAI environment can pull in incompatible package versions and break the application.

If a node pack ships a `requirements.txt` or `pyproject.toml`, you'll see a warning toast after installation. Install the dependencies yourself by following the instructions in the node pack's documentation (typically `pip install -r requirements.txt` from inside an activated InvokeAI environment, but check the pack's README first). After installing, click the **Reload** button so the new dependencies take effect.

### Security Warning

Custom nodes execute arbitrary Python code on your system. **Only install node packs from authors you trust.** Malicious nodes could harm your system or compromise your data.

## Managing Installed Nodes

The left panel shows all installed node packs with:

- **Pack name**
- **Number of nodes** provided
- **Individual node types** as badges
- **File path** on disk

### Reloading Nodes

Click the **Reload** button to re-scan the nodes directory. This picks up any node packs that were manually added to the directory without using the installer.

### Uninstalling a Node Pack

Click the **Uninstall** button on any node pack. This will:

- Remove the node pack directory
- Unregister the nodes from the system immediately
- Remove any workflows that were imported from the pack
- Update the workflow editor so the nodes are no longer available

No restart is required.

## Scan Folder Tab

The **Scan Folder** tab shows the location of your nodes directory. Node packs placed there manually (e.g. via `git clone`) are automatically detected at startup. Use the **Reload** button to detect newly added packs without restarting.

## Troubleshooting

### Node pack fails to install

- Verify the Git URL is correct and accessible
- Check that the repository contains an `__init__.py` file at the top level
- Review the Install Log for error details

### Nodes don't appear after install

- Click the **Reload** button
- Check that the node pack's `__init__.py` imports its node classes
- Check the server console for error messages

### Workflows show errors after uninstalling

If you have user-created workflows that reference nodes from an uninstalled pack, those workflows will show errors for the missing node types. Reinstall the pack or remove the affected nodes from the workflow.
