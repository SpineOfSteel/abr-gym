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
import { AbrAlgorithm } from '../algo/interface';
import { ThrpPredictorGetter } from '../algo/getters';
import { Decision } from '../common/data';
import { logging } from '../common/logger';
var logger = logging('Festive');
var diminuation_factor = 0.85;
var alpha = 12;
var horizon = 5;
var switchUpThreshold = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/**
 * Festive(https://dl.acm.org/doi/10.1145/2413176.2413189) is a rate-based approach that uses a
 * windowed harmonic mean for bandwidth estimation. The code has mostly been ported from the Pensive
 * repository: https://github.com/hongzimao/pensieve.
 */
var Festive = /** @class */ (function (_super) {
    __extends(Festive, _super);
    function Festive(video) {
        var _this = _super.call(this) || this;
        _this.bitrateArray = video.bitrateArray;
        _this.n = _this.bitrateArray.length;
        _this.bandwidth = new ThrpPredictorGetter();
        _this.qualityLog = { 0: 0 };
        _this.prevQuality = 0;
        _this.lastIndex = 0;
        _this.switchUpCount = 0;
        return _this;
    }
    /**
     * Selects a `quality` for a `bitrate`. Select the first quality for which the associated
     * bitrateArray value stays below the `bitrate`.
     */
    Festive.prototype.selectQuality = function (bitrate) {
        var quality = this.n;
        for (var i = this.n - 1; i >= 0; i--) {
            quality = i;
            if (bitrate >= this.bitrateArray[i]) {
                break;
            }
        }
        return quality;
    };
    /**
     * Given the current quality `b` and a reference quality `b_ref` and a bandwidth prediction
     * `w` compute the efficiency score as presented in the paper.
     */
    Festive.prototype.getEfficiencyScore = function (b, b_ref, w) {
        return Math.abs(this.bitrateArray[b]
            / Math.min(w, this.bitrateArray[b_ref]) - 1);
    };
    /**
     * Given the current candate quality `b`, a reference quality `b_ref`, a candidate quality `b_cur`
     * compute the stability score as presented in the paper.
     */
    Festive.prototype.getStabilityScore = function (b, b_ref, b_cur) {
        var score = 0, changes = 0;
        if (this.lastIndex >= 1) {
            var start = Math.max(0, this.lastIndex + 1 - horizon);
            var end = this.lastIndex - 1;
            for (var i = start; i <= end; i++) {
                if (this.qualityLog[i] != this.qualityLog[i + 1]) {
                    changes++;
                }
            }
        }
        if (b != b_cur) {
            changes += 1;
        }
        score = Math.pow(2, changes);
        return score;
    };
    Festive.prototype.getCombinedScore = function (b, b_ref, b_cur, w) {
        var stabilityScore = this.getStabilityScore(b, b_ref, b_cur);
        var efficiencyScore = this.getEfficiencyScore(b, b_ref, w);
        return stabilityScore + alpha * efficiencyScore;
    };
    /**
     * The decision is taken as follows:
     *   - compute a target quality: b_target based on the future bandwidth prediction
     *   - compute the new reference target quality: b_ref which is limited by number of upward quality
     *     changes
     *   - keep or modify the quality based on the computed combined scores for the qualities
     */
    Festive.prototype.getDecision = function (metrics, index, timestamp) {
        this.bandwidth.update(metrics, this.requests);
        var bwPrediction = this.bandwidth.value;
        // compute b_target
        var b_target = this.selectQuality(diminuation_factor * bwPrediction);
        // compute b_ref
        var b_cur = this.prevQuality;
        var b_ref = 0;
        if (b_target > b_cur) {
            this.switchUpCount = this.switchUpCount + 1;
            if (this.switchUpCount > switchUpThreshold[b_cur]) {
                b_ref = b_cur + 1;
            }
            else {
                b_ref = b_cur;
            }
        }
        else if (b_target < b_cur) {
            b_ref = b_cur - 1;
            this.switchUpCount = 0;
        }
        else {
            b_ref = b_cur;
            this.switchUpCount = 0;
        }
        // delayed update
        var quality = 0;
        if (b_ref != b_cur) {
            // compute scores
            var score_cur = this.getCombinedScore(b_cur, b_ref, b_cur, bwPrediction);
            var score_ref = this.getCombinedScore(b_ref, b_ref, b_cur, bwPrediction);
            logger.log("score cur ".concat(b_cur, " -> ").concat(score_cur), "score ref ".concat(b_ref, " -> ").concat(score_ref));
            if (score_cur <= score_ref) {
                quality = b_cur;
            }
            else {
                quality = b_ref;
                if (quality > b_cur) {
                    this.switchUpCount = 0;
                }
            }
        }
        else {
            quality = b_cur;
        }
        // log relevant info for festive
        logger.log("quality ".concat(quality), "b_target ".concat(b_target));
        // update quality log
        this.qualityLog[index] = this.bitrateArray[quality];
        this.prevQuality = quality;
        this.lastIndex = index;
        return new Decision(index, quality, timestamp);
    };
    return Festive;
}(AbrAlgorithm));
export { Festive };
//# sourceMappingURL=festive.js.map