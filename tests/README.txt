The nodes tests need to run before the others in order to avoid a race
condition involving fixture initialization. Please see
https://discord.com/channels/1020123559063990373/1156089584808120382/1156802853323673620
for an explanation.

For this reason, the subtests are grouped into alphabetically-ordered
folders. Do not use numeric prefixes (e.g. 00_nodes) because this
breaks python's import system.
