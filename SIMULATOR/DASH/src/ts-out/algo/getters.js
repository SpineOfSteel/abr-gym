var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __values = (this && this.__values) || function(o) {
    var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
    if (m) return m.call(o);
    if (o && typeof o.length === "number") return {
        next: function () {
            if (o && i >= o.length) o = void 0;
            return { value: o && o[i++], done: !o };
        }
    };
    throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
};
import { MetricGetter } from '../algo/interface';
import { Value, Segment } from '../common/data';
import { logging } from '../common/logger';
import { timestamp } from '../common/time';
var logger = logging('Getters');
var defaultThrp = 0;
/**
 * Getter that computes the latest timestamped rebuffer time in ms.
 */
var RebufferTimeGetter = /** @class */ (function (_super) {
    __extends(RebufferTimeGetter, _super);
    function RebufferTimeGetter() {
        var _this = _super.call(this) || this;
        _this.smallestPlayerTime = null;
        _this.biggestPlayerTime = new Value(0).withTimestamp(0);
        return _this;
    }
    RebufferTimeGetter.prototype.update = function (metrics) {
        var e_1, _a;
        try {
            for (var _b = __values(metrics.playerTime), _c = _b.next(); !_c.done; _c = _b.next()) {
                var playerTime = _c.value;
                if (this.smallestPlayerTime === null) {
                    this.smallestPlayerTime = playerTime;
                }
                else if (this.smallestPlayerTime.timestamp > playerTime.timestamp) {
                    this.smallestPlayerTime = playerTime;
                }
                if (this.biggestPlayerTime.timestamp < playerTime.timestamp) {
                    this.biggestPlayerTime = playerTime;
                }
            }
        }
        catch (e_1_1) { e_1 = { error: e_1_1 }; }
        finally {
            try {
                if (_c && !_c.done && (_a = _b.return)) _a.call(_b);
            }
            finally { if (e_1) throw e_1.error; }
        }
    };
    Object.defineProperty(RebufferTimeGetter.prototype, "value", {
        get: function () {
            if (this.smallestPlayerTime !== null) {
                var ts_diff = this.biggestPlayerTime.timestamp - this.smallestPlayerTime.timestamp;
                var value_diff = this.biggestPlayerTime.value - this.smallestPlayerTime.value;
                var value = ts_diff - value_diff;
                if (value < 0) {
                    return 0;
                }
                else {
                    return value;
                }
            }
            return 0;
        },
        enumerable: false,
        configurable: true
    });
    return RebufferTimeGetter;
}(MetricGetter));
export { RebufferTimeGetter };
/**
 * Getter that computes the latest timestamped buffer level.
 */
var BufferLevelGetter = /** @class */ (function (_super) {
    __extends(BufferLevelGetter, _super);
    function BufferLevelGetter() {
        var _this = _super.call(this) || this;
        _this.lastBufferLevel = new Value(0).withTimestamp(0);
        return _this;
    }
    BufferLevelGetter.prototype.update = function (metrics) {
        this.lastBufferLevel = metrics.bufferLevel.reduce(function (a, b) { return a.timestamp < b.timestamp ? b : a; }, this.lastBufferLevel);
    };
    Object.defineProperty(BufferLevelGetter.prototype, "value", {
        get: function () {
            var value_at_timestmap = this.lastBufferLevel.value;
            var consumed = timestamp(new Date()) - this.lastBufferLevel.timestamp;
            return Math.max(value_at_timestmap - consumed, 0);
        },
        enumerable: false,
        configurable: true
    });
    return BufferLevelGetter;
}(MetricGetter));
export { BufferLevelGetter };
/**
 * Abstract getter that computes:
 *  - requests: a list of all the XMLHttpRequests made so far
 *  - segments: a list containing the download state of all the Segments encoutered so far
 *  - lastSegment: the index of the latest segments for which a download was started
 *  - horizon: a horizon constant used by throuput-based estimators to be derived from this class
 */
var ThrpGetter = /** @class */ (function (_super) {
    __extends(ThrpGetter, _super);
    function ThrpGetter() {
        var _this = _super.call(this) || this;
        _this.segments = { 1: new Segment().withTimestamp(0) };
        _this.requests = [];
        _this.lastSegment = 0;
        _this.horizon = 5;
        return _this;
    }
    ThrpGetter.prototype.update = function (metrics, requests) {
        var e_2, _a;
        try {
            for (var _b = __values(metrics.segments), _c = _b.next(); !_c.done; _c = _b.next()) {
                var segment = _c.value;
                if (this.segments[segment.index] === undefined) {
                    this.segments[segment.index] = segment;
                }
                if (segment.index > this.lastSegment) {
                    this.lastSegment = segment.index;
                }
            }
        }
        catch (e_2_1) { e_2 = { error: e_2_1 }; }
        finally {
            try {
                if (_c && !_c.done && (_a = _b.return)) _a.call(_b);
            }
            finally { if (e_2) throw e_2.error; }
        }
        this.requests = requests;
    };
    return ThrpGetter;
}(MetricGetter));
export { ThrpGetter };
/**
 * Throughput predictor that computes the throughput based on the harmonic mean of the observed
 * throughput of the last segments in the horizon. The implementation coincides with the take on
 * RobustMpc(https://users.ece.cmu.edu/~vsekar/papers/sigcomm15_mpcdash.pdf) as a baseline the
 * repository: https://github.com/hongzimao/pensieve.
 */
var ThrpPredictorGetter = /** @class */ (function (_super) {
    __extends(ThrpPredictorGetter, _super);
    function ThrpPredictorGetter() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Object.defineProperty(ThrpPredictorGetter.prototype, "value", {
        get: function () {
            if (this.lastSegment < 2) {
                return 0;
            }
            var totalSum = 0;
            var totalTime = 0;
            var minIndex = Math.max(this.lastSegment - this.horizon + 1, 2);
            for (var index = this.lastSegment; index >= minIndex; --index) {
                var time = 0;
                // Race condition -- if we get no metrics, then we need to pass some defaultThrp
                // -- in this case we pass 0
                if (this.segments[index - 1] !== undefined) {
                    time = this.segments[index].timestamp - this.segments[index - 1].timestamp;
                }
                var size = this.requests[index - 2].response.byteLength * 8;
                totalSum += time * time / size;
                totalTime += time;
            }
            if (totalSum == 0) {
                return defaultThrp;
            }
            return totalTime / totalSum;
        },
        enumerable: false,
        configurable: true
    });
    return ThrpPredictorGetter;
}(ThrpGetter));
export { ThrpPredictorGetter };
/**
 * Computes the throughput of the last segment.
 */
var LastThrpGetter = /** @class */ (function (_super) {
    __extends(LastThrpGetter, _super);
    function LastThrpGetter() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Object.defineProperty(LastThrpGetter.prototype, "value", {
        get: function () {
            if (this.lastSegment < 2) {
                return 0;
            }
            var index = this.lastSegment;
            var time = 0;
            // Race condition -- if we get no metrics, then we need to pass some defaultThrp
            // -- in this case we pass 0
            if (this.segments[index - 1] !== undefined) {
                time = this.segments[index].timestamp - this.segments[index - 1].timestamp;
            }
            var size = this.requests[index - 2].response.byteLength * 8;
            if (time == 0) {
                return defaultThrp;
            }
            return size / time;
        },
        enumerable: false,
        configurable: true
    });
    return LastThrpGetter;
}(ThrpGetter));
export { LastThrpGetter };
/**
 * Computes the diffrence between timestmaps for the start of download for the last 2 segments.
 */
var LastFetchTimeGetter = /** @class */ (function (_super) {
    __extends(LastFetchTimeGetter, _super);
    function LastFetchTimeGetter() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Object.defineProperty(LastFetchTimeGetter.prototype, "value", {
        get: function () {
            if (this.lastSegment < 2) {
                return 0;
            }
            var index = this.lastSegment;
            var time = 0;
            // Race condition -- if we get no metrics, then we need to pass 0
            if (this.segments[index - 1] !== undefined) {
                time = this.segments[index].timestamp - this.segments[index - 1].timestamp;
            }
            return time;
        },
        enumerable: false,
        configurable: true
    });
    return LastFetchTimeGetter;
}(ThrpGetter));
export { LastFetchTimeGetter };
//# sourceMappingURL=getters.js.map