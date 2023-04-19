# Build the container
From the docker directory run these commands:
```bash
cd ..
./docker/build.sh
```

# Run the contianer
Run the container with the following commands:

***Note***: Replace `DB_USER_NAME`, `DB_PASSWORD`, `DB_SERVER`, `DB_DATABASE` and the `IMAGE_NAME` with appropriate values

```bash
docker run -it \
    --cap-add=SYS_ADMIN \
    --device=/dev/fuse \
    --security-opt apparmor:unconfined \
    --publish=9090:9090 \
    --env db_dialect=mysql+pymysql \
    --env db_user=DB_USER_NAME \
    --env db_password=DB_PASSWORD \
    --env db_server=DB_SERVER \
    --env db_database=DB_DATABASE \
    IMAGE_NAME
```