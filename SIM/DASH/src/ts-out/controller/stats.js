import { Metrics } from '../component/stats';
import { logging } from '../common/logger';
var logger = logging('StatsController');
/**
 * Single source of truth for all Metrics updates.
 */
var StatsController = /** @class */ (function () {
    function StatsController() {
        this._metrics = new Metrics();
    }
    /**
     * Discard all information before timestamp, inclusively.
     */
    StatsController.prototype.advance = function (timestamp) {
        var filter = function (value) { return value.timestamp > timestamp; };
        this._metrics = new Metrics().withMetrics(this._metrics, filter);
    };
    /**
     * Get all metrics until the timestamp, inclusively. If a timestamp was not
     * provided, return all the metrics. The metrics are return in a new Metrics object.
     */
    StatsController.prototype.getMetrics = function (timestamp) {
        if (timestamp === undefined) {
            return new Metrics().withMetrics(this._metrics).sorted();
        }
        else {
            var filter = function (value) { return value.timestamp <= timestamp; };
            return new Metrics().withMetrics(this._metrics, filter).sorted();
        }
    };
    Object.defineProperty(StatsController.prototype, "metrics", {
        /**
         * Return all the metrics.
         */
        get: function () {
            return this.getMetrics();
        },
        enumerable: false,
        configurable: true
    });
    /**
     * Add a new Metrics object as an update over the current held metrics.
     */
    StatsController.prototype.addMetrics = function (metrics) {
        this._metrics.withMetrics(metrics);
    };
    return StatsController;
}());
export { StatsController };
//# sourceMappingURL=stats.js.map