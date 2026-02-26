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
/**
 * Metadata assciated with a Segment:
 *  - start_time: start time of the segment in the video in seconds
 *  - VMAF: value between 0 and 100 representing the perceptual quality
 *  - size: size of the segment in bytes
 */
var SegmentInfo = /** @class */ (function () {
    function SegmentInfo(obj) {
        if (obj["start_time"] === undefined ||
            obj["vmaf"] === undefined ||
            obj["size"] === undefined) {
            throw new TypeError("Wrong segment info format ".concat(obj));
        }
        this.start_time = obj["start_time"];
        this.vmaf = obj["vmaf"];
        this.size = obj["size"];
    }
    return SegmentInfo;
}());
export { SegmentInfo };
/**
 * Collection of video-level metadata:
 *  - the available segment bands
 *  - the segment info for each segment from each band
 */
var VideoInfo = /** @class */ (function () {
    function VideoInfo(config) {
        var e_1, _a, e_2, _b, e_3, _c;
        // save the bitrates
        this.bitrates = [];
        try {
            for (var _d = __values(config.video_paths), _e = _d.next(); !_e.done; _e = _d.next()) {
                var conf = _e.value;
                this.bitrates.push(conf.quality);
            }
        }
        catch (e_1_1) { e_1 = { error: e_1_1 }; }
        finally {
            try {
                if (_e && !_e.done && (_a = _d.return)) _a.call(_d);
            }
            finally { if (e_1) throw e_1.error; }
        }
        this.bitrates.sort(function (a, b) { return a - b; });
        // save video information
        this.info = {};
        try {
            for (var _f = __values(config.video_paths), _g = _f.next(); !_g.done; _g = _f.next()) {
                var conf = _g.value;
                var segments = [];
                try {
                    for (var _h = (e_3 = void 0, __values(conf.info)), _j = _h.next(); !_j.done; _j = _h.next()) {
                        var raw_segment_info = _j.value;
                        segments.push(new SegmentInfo(raw_segment_info));
                    }
                }
                catch (e_3_1) { e_3 = { error: e_3_1 }; }
                finally {
                    try {
                        if (_j && !_j.done && (_c = _h.return)) _c.call(_h);
                    }
                    finally { if (e_3) throw e_3.error; }
                }
                this.info[conf.quality] = segments;
            }
        }
        catch (e_2_1) { e_2 = { error: e_2_1 }; }
        finally {
            try {
                if (_g && !_g.done && (_b = _f.return)) _b.call(_f);
            }
            finally { if (e_2) throw e_2.error; }
        }
    }
    Object.defineProperty(VideoInfo.prototype, "bitrateArray", {
        get: function () {
            return this.bitrates;
        },
        enumerable: false,
        configurable: true
    });
    return VideoInfo;
}());
export { VideoInfo };
//# sourceMappingURL=video.js.map