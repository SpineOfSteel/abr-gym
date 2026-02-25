/**
 * AbrAlgorithm interface
 */
var AbrAlgorithm = /** @class */ (function () {
    function AbrAlgorithm() {
        this.requests = [];
    }
    AbrAlgorithm.prototype.newRequest = function (ctx) {
        this.requests.push(ctx);
    };
    return AbrAlgorithm;
}());
export { AbrAlgorithm };
/**
 * MetricGetter interface: can be used to implement a specific derivation of a particular metric
 * given the previous HTTP requests and front-end metrics.
 */
var MetricGetter = /** @class */ (function () {
    function MetricGetter() {
    }
    return MetricGetter;
}());
export { MetricGetter };
//# sourceMappingURL=interface.js.map