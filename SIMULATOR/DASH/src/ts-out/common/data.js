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
import { timestamp as create_timestamp } from '../common/time';
export var SEGMENT_STATE = {
    DOWNLOADED: 'downloaded',
    PROGRESS: 'progress',
    LOADING: 'loading',
};
var Piece = /** @class */ (function () {
    function Piece() {
    }
    return Piece;
}());
export { Piece };
/**
 * Timestamped real number.
 */
var Value = /** @class */ (function () {
    function Value(value) {
        this._value = value;
        this._timestamp = create_timestamp(new Date());
    }
    Value.prototype.withTimestamp = function (timestamp) {
        this._timestamp = timestamp;
        return this;
    };
    Object.defineProperty(Value.prototype, "value", {
        get: function () {
            return this._value;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Value.prototype, "timestamp", {
        get: function () {
            return this._timestamp;
        },
        enumerable: false,
        configurable: true
    });
    Value.prototype.serialize = function () {
        return {
            "value": this.value,
            "timestamp": this.timestamp,
        };
    };
    return Value;
}());
export { Value };
/**
 * Data-representation for a decision taken based on metrics received until timestamp `timestamp`
 * for piece number `index` for downloading at quality `quality`.
 */
var Decision = /** @class */ (function (_super) {
    __extends(Decision, _super);
    function Decision(index, quality, timestamp) {
        var _this = _super.call(this) || this;
        _this._index = index;
        _this._quality = quality;
        _this._timestamp = timestamp;
        return _this;
    }
    Object.defineProperty(Decision.prototype, "index", {
        get: function () {
            return this._index;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Decision.prototype, "quality", {
        get: function () {
            return this._quality;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Decision.prototype, "timestamp", {
        get: function () {
            return this._timestamp;
        },
        enumerable: false,
        configurable: true
    });
    return Decision;
}(Piece));
export { Decision };
/**
 * Segment representation used for serialization in communication with the back-end and experiment
 * coordinator. Contains both metadata(quality, index, state) related to the segment together with the
 * state of the download process(`loaded` bytes out of `total`).
 *
 * Can be serialized via the `serialize` function.
 **/
var Segment = /** @class */ (function (_super) {
    __extends(Segment, _super);
    function Segment() {
        var _this = _super.call(this) || this;
        _this._timestamp = create_timestamp(new Date());
        return _this;
    }
    Segment.prototype.withLoaded = function (loaded) {
        this._loaded = loaded;
        return this;
    };
    Segment.prototype.withTotal = function (total) {
        this._total = total;
        return this;
    };
    Segment.prototype.withTimestamp = function (timestamp) {
        this._timestamp = timestamp;
        return this;
    };
    Segment.prototype.withQuality = function (quality) {
        this._quality = quality;
        return this;
    };
    Segment.prototype.withState = function (state) {
        this._state = state;
        return this;
    };
    Segment.prototype.withIndex = function (index) {
        this._index = index;
        return this;
    };
    Segment.prototype.withStartTime = function (startTime, duration) {
        // segments start from 1
        this._index = Math.round(startTime / duration) + 1;
        return this;
    };
    Object.defineProperty(Segment.prototype, "total", {
        get: function () {
            return this._total;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Segment.prototype, "loaded", {
        get: function () {
            return this._loaded;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Segment.prototype, "timestamp", {
        get: function () {
            return this._timestamp;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Segment.prototype, "index", {
        get: function () {
            return this._index;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Segment.prototype, "quality", {
        get: function () {
            return this._quality;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Segment.prototype, "state", {
        get: function () {
            return this._state;
        },
        enumerable: false,
        configurable: true
    });
    /**
     * Serialize the segment. If the `full` value is set, the `quality` of the segment will be
     * included in the JSON serialization.
     */
    Segment.prototype.serialize = function (full) {
        var ret = {};
        if (this.state == SEGMENT_STATE.LOADING || this.state == SEGMENT_STATE.DOWNLOADED) {
            ret = {
                "index": this.index,
                "state": this.state,
                "timestamp": this.timestamp,
            };
        }
        else if (this.state == SEGMENT_STATE.PROGRESS) {
            ret = {
                "index": this.index,
                "state": this.state,
                "timestamp": this.timestamp,
                "loaded": this.loaded,
                "total": this.total,
            };
        }
        else {
            throw new RangeError("Unrecognized segment state ".concat(this.state));
        }
        if (full) {
            ret = Object.assign(ret, {
                "quality": this.quality,
            });
        }
        return ret;
    };
    return Segment;
}(Piece));
export { Segment };
//# sourceMappingURL=data.js.map