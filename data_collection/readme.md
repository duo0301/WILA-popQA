# Data Collection readme

In order to address the challenges we encountered with the Wikidata Dump and Blazegraph engine, we propose using qEndpoint [1] with the HDT format [2] of Wikidata. This approach allows us to query data much faster while retrieving only the necessary data efficiently.

## Quick Setup

The fastest way to test and deploy this solution is to use the provided Docker Image [3] for qEndpoint. 
It simplifies the setup and integration process, enabling us to start querying the HDT files almost immediately.

Everyone should be able query Wikidata localy. 
We are using "wikidata-truthy-28.10.23.hdt" (the truthy version of wikidata).

setup sparql endpoint (qEndpoint)

```bash
docker run -p 1234:1234 --name qendpoint-wikidata --env MEM_SIZE=6G qacompany/qendpoint-wikidata
```

Quick test with the following query :

```bash
curl -H 'Accept: application/sparql-results+json' localhost:1234/api/endpoint/sparql --data-urlencode 'query=select * where{ ?s ?p ?o } limit 2'
```

If everything is working, you should get a json response with 2 triples from Wikidata.
You will also have access to the endpoint at http://localhost:1234/ and you can test your queries there.

## Dataset construction process

Principal commands

```pyhton
python get_data.py
python getEntities_pointsSys.py
python property_coverage_analyzer.py
python Coverage_setprop_per_langauge.py
python3 filter_ids.py
```

## Other datasets available

Pre-generated HDT files for Wikidata are available on [5]. Examples include:

- wikidata-truthy-28.10.23.hdt (21-Dec-2023)
- wikidata_all.hdt (08-Sep-2024)
- wikidata_truthy.hdt (21-Dec-2023)

Corresponding .index files (e.g., wikidata_all.hdt.index.v1-1) are also available to speed up queries.


## References

[1] [qEndpoint GitHub](https://github.com/the-qa-company/qEndpoint)
[2] [HDT Datasets](https://www.rdfhdt.org/datasets/)
[3] [Docker Image for qEndpoint](https://github.com/the-qa-company/qEndpoint?tab=readme-ov-file#docker-image)
[4] [HDT Technical Format](https://www.rdfhdt.org/hdt-binary-format/)
[5] [KG Datasets with HDT Files](https://qanswer-svc4.univ-st-etienne.fr/)
[6] [rdf2hdt Tool](https://www.rdfhdt.org/manual-of-the-c-hdt-library/)
[7] [Instruction to set qendpoint docker image for Wikidata](https://github.com/the-qa-company/qEndpoint?tab=readme-ov-file#qacompanyqendpoint-wikidata)
[8] [qEndpoint on docker hub](https://hub.docker.com/r/qacompany/qendpoint-wikidata)