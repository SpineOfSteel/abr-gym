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
var __read = (this && this.__read) || function (o, n) {
    var m = typeof Symbol === "function" && o[Symbol.iterator];
    if (!m) return o;
    var i = m.call(o), r, ar = [], e;
    try {
        while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
    }
    catch (error) { e = { error: error }; }
    finally {
        try {
            if (r && !r.done && (m = i["return"])) m.call(i);
        }
        finally { if (e) throw e.error; }
    }
    return ar;
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
import { timestamp } from '../common/time';
import { Value, Segment, SEGMENT_STATE } from '../common/data';
import { default as stringify } from 'json-stable-stringify';
import { logging } from '../common/logger';
var logger = logging('Metrics');
var TICK_INTERVAL_MS = 100;
var MEDIA_TYPE = 'video';
function pushDefault(context, value) {
    if (context === undefined) {
        context = [];
    }
    context.push(value);
}
/**
 * Serializable object that contains all front-end metrics:
 *   - number of dropped frames at each timestamp since last metrics update
 *   - the player time(in seconds) at each timestamp since last metrics update
 *   - the buffer level(in milliseconds) at each timestamp since last metrics update
 *   - the list of new segments(including their download state) since last metrics update
 */
var Metrics = /** @class */ (function () {
    function Metrics(raw_metrics) {
        this.clear();
        if (raw_metrics !== undefined) {
            var segment = new Segment()
                .withStartTime(raw_metrics.scheduling.startTime, raw_metrics.scheduling.duration)
                .withState(SEGMENT_STATE.LOADING)
                .withTimestamp(timestamp(raw_metrics.scheduling.t))
                .withQuality(raw_metrics.scheduling.quality);
            this.withSegment(segment);
            this.withDroppedFrames(new Value(raw_metrics.dropped.droppedFrames));
            this.withPlayerTime(new Value(Math.round(raw_metrics.info.time * 1000)));
            this.withBufferLevel(new Value(Math.round(raw_metrics.buffer_level * 1000)));
        }
    }
    Metrics.prototype._apply = function (builder, array, filter) {
        var e_1, _a;
        if (filter !== undefined) {
            array = array.filter(filter);
        }
        try {
            for (var array_1 = __values(array), array_1_1 = array_1.next(); !array_1_1.done; array_1_1 = array_1.next()) {
                var value = array_1_1.value;
                this[builder](value);
            }
        }
        catch (e_1_1) { e_1 = { error: e_1_1 }; }
        finally {
            try {
                if (array_1_1 && !array_1_1.done && (_a = array_1.return)) _a.call(array_1);
            }
            finally { if (e_1) throw e_1.error; }
        }
        return this;
    };
    /**
     * Builder-patter for adding new Metrics filtered by a filter applicable for
     * timestamped values.
     */
    Metrics.prototype.withMetrics = function (metrics, filter) {
        return this
            ._apply('withDroppedFrames', metrics.droppedFrames, filter)
            ._apply('withPlayerTime', metrics.playerTime, filter)
            ._apply('withBufferLevel', metrics.bufferLevel, filter)
            ._apply('withSegment', metrics.segments, filter);
    };
    /**
     * Drop all metrics.
     */
    Metrics.prototype.clear = function () {
        this._droppedFrames = [];
        this._bufferLevel = [];
        this._playerTime = [];
        this._segments = [];
        return this;
    };
    /**
     * Serialize metrics; if noProgress is true, then don't include the segment serialization
     */
    Metrics.prototype.serialize = function (noProgress) {
        if (noProgress === void 0) { noProgress = false; }
        // @ts-ignore: unsafe to use stringify, then JSON.parse
        var unique = function (arr) { return __spreadArray([], __read(new Set(arr.map(stringify))), false).map(JSON.parse); };
        // @ts-ignore
        var transform = function (arr) { return unique(arr.map(function (x) { return x.serialize(noProgress); })); };
        var cmp = function (a, b) { return a.timestamp - b.timestamp; };
        var prepareSegments = function (segments) {
            var groupBy = function (xs, map) { return xs.reduce(function (rv, x) {
                (rv[map(x)] = rv[map(x)] || []).push(x);
                return rv;
            }, {}); };
            var statelessFilter = function (array, filter) { return array.reduce(function (acc, v) {
                if (filter(v))
                    acc.push(v);
                return acc;
            }, []); };
            var prepareLoading = function (segments) {
                var out = [];
                if (noProgress) {
                    return out;
                }
                var grouped = groupBy(segments, function (segment) { return segment.index; });
                Object.keys(grouped).forEach(function (index) {
                    var segment = grouped[index].sort(cmp).slice(-1)[0];
                    out.push(segment);
                });
                return out;
            };
            return transform(statelessFilter(segments, function (s) { return s.state != SEGMENT_STATE.PROGRESS; }).concat(prepareLoading(statelessFilter(segments, function (s) { return s.state == SEGMENT_STATE.PROGRESS; })))).sort(cmp);
        };
        return {
            "droppedFrames": transform(this._droppedFrames).sort(cmp),
            "playerTime": transform(this._playerTime).sort(cmp),
            "bufferLevel": transform(this._bufferLevel).sort(cmp),
            "segments": prepareSegments(this._segments),
        };
    };
    /**
     * Sorts metrics by timestamp; mutating the current object.
     */
    Metrics.prototype.sorted = function () {
        var cmp = function (a, b) { return a.timestamp - b.timestamp; };
        this._droppedFrames.sort(cmp);
        this._playerTime.sort(cmp);
        this._bufferLevel.sort(cmp);
        this._segments.sort(cmp);
        return this;
    };
    /**
     * Builder for adding a *single* dropped frames metric.
     */
    Metrics.prototype.withDroppedFrames = function (droppedFrames) {
        this._droppedFrames.push(droppedFrames);
        return this;
    };
    /**
     * Builder for adding a *single* buffer level value.
     */
    Metrics.prototype.withBufferLevel = function (bufferLevel) {
        this._bufferLevel.push(bufferLevel);
        return this;
    };
    /**
     * Builder for adding a *single* player time value.
     */
    Metrics.prototype.withPlayerTime = function (playerTime) {
        this._playerTime.push(playerTime);
        return this;
    };
    /**
     * Builder for adding a *single* segment; requires the segment index
     * to be an integer number.
     */
    Metrics.prototype.withSegment = function (segment) {
        if (!isNaN(segment.index)) {
            this._segments.push(segment);
        }
        return this;
    };
    Object.defineProperty(Metrics.prototype, "segments", {
        /******************
         * Getters below. *
         ******************/
        get: function () {
            return this._segments;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Metrics.prototype, "bufferLevel", {
        get: function () {
            return this._bufferLevel;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Metrics.prototype, "playerTime", {
        get: function () {
            return this._playerTime;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Metrics.prototype, "droppedFrames", {
        get: function () {
            return this._droppedFrames;
        },
        enumerable: false,
        configurable: true
    });
    return Metrics;
}());
export { Metrics };
/**
 * Class used for allowing automatic Metrics updates.
 */
var StatsTracker = /** @class */ (function () {
    function StatsTracker(player) {
        this.player = player;
        this.callbacks = [];
    }
    /**
     * Start the automatic metrics update. A tick will be made every TICK_INTERVAL_MS.
     */
    StatsTracker.prototype.start = function () {
        var _this = this;
        this.getMetrics();
        setInterval(function () {
            _this.getMetrics();
        }, TICK_INTERVAL_MS);
    };
    /**
     * The tracker updates the Metrics from the DASH player. In case the withCallbacks
     * flag is on, all metrics callbacks are apllied to the metrics object.
     *
     * The `getMetrics` function is exposed as we might want to call it explictly after significant
     * modifications of the DASH player(e.g. new segment request). The callbacks *mutate* the metrics.
     */
    StatsTracker.prototype.getMetrics = function (withCallbacks) {
        var e_2, _a;
        if (withCallbacks === void 0) { withCallbacks = true; }
        var metrics_wrapper, metrics;
        try {
            metrics_wrapper = this.player.getDashMetrics();
            metrics = this.tick(metrics_wrapper);
        }
        catch (err) {
            metrics = new Metrics();
        }
        if (withCallbacks) {
            try {
                for (var _b = __values(this.callbacks), _c = _b.next(); !_c.done; _c = _b.next()) {
                    var callback = _c.value;
                    callback(metrics);
                }
            }
            catch (e_2_1) { e_2 = { error: e_2_1 }; }
            finally {
                try {
                    if (_c && !_c.done && (_a = _b.return)) _a.call(_b);
                }
                finally { if (e_2) throw e_2.error; }
            }
        }
    };
    /**
     * Internal function: the tick returns a single mutable metrics object.
     *
     * Given a *metrics_wrapper* provided by the DASH player's method `getDashMetrics`,
     * return a Metrics object.
     */
    StatsTracker.prototype.tick = function (metrics_wrapper) {
        var execute = function (func) {
            var args = [];
            for (var _i = 1; _i < arguments.length; _i++) {
                args[_i - 1] = arguments[_i];
            }
            try {
                return func.apply(void 0, __spreadArray([], __read(args), false));
            }
            catch (err) {
                return null;
            }
        };
        var raw_metrics = {
            'info': execute(metrics_wrapper.getCurrentDVRInfo, MEDIA_TYPE),
            'dropped': execute(metrics_wrapper.getCurrentDroppedFrames),
            'switch': execute(metrics_wrapper.getCurrentRepresentationSwitch, MEDIA_TYPE, true),
            'scheduling': execute(metrics_wrapper.getCurrentSchedulingInfo, MEDIA_TYPE),
            'buffer_level': execute(metrics_wrapper.getCurrentBufferLevel, MEDIA_TYPE),
        };
        var metrics = new Metrics(raw_metrics);
        return metrics;
    };
    /**
     * Register a callback to be called during each getMetrics method.
     */
    StatsTracker.prototype.registerCallback = function (callback) {
        this.callbacks.push(callback);
    };
    return StatsTracker;
}());
export { StatsTracker };
//# sourceMappingURL=stats.js.map