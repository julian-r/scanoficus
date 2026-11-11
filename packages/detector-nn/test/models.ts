import { DOCQUADNET } from '../src/bundle-docquadnet.ts';

/** File URLs of the packaged round-1 model (tests read them with `readFileSync`). */
export const DOCQUADNET_GRAPH_URL: URL = DOCQUADNET.main.graphUrl as URL;
export const DOCQUADNET_WEIGHTS_URL: URL = DOCQUADNET.main.weightsUrl as URL;
