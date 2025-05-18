### MinIO Service Key

---

To be able to use the SSPCloud _MinIO_ storage (AWS S3-type service) at all times, both to avoid key expiration and renewal issues, and to use it from outside, we can generate a service key that does not expire and will be used by everyone.

To do this, we can use the MinIO *mc* client (`mc.exe`). To create the service key, edit a `policy.json` file with the correct project/bucket name, then, having properly configured the environment variables with your personal access (only used to create the key), execute for example:

```bash
mc admin accesskey create s3 --access-key="statapp-segmedic" --secret-key="hiMITEpoWnigHtERylvANaL" --policy="policy.json"
```

You should normally get an output of the form:

```yml
$ mc admin [...]
Access Key: statapp-segmedic
Secret Key: hiMITEpoWnigHtERylvANaL
Expiration: NONE
Name:
Description:
```

These variables can be used in the project's `.env` file.

#### Learn more:

- MinIO Documentation: [here](https://min.io/docs/minio/linux/reference/minio-mc.html)
- SSPCloud Storage Documentation: [here](https://docs.sspcloud.fr/content/storage.html)
