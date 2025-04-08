import { r2rClient } from "r2r-js";
const client = new r2rClient("https://api.r2r.ai", true);

client.documents
  .create({ file: "test" })
  .then(console.log)
  .catch(console.error);
