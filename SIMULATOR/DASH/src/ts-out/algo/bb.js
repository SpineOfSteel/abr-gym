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
import { BufferLevelGetter } from '../algo/getters';
import { Decision } from '../common/data';
import { logging } from '../common/logger';
var logger = logging('BB');
var SECOND = 1000;
var reservoir = 5 * SECOND;
var cushion = 10 * SECOND;
/**
 * Buffer-based ABR algorithm
 *
 * If the buffer is below the `reservoir`, we use the smallest available quality level.
 * If the buffer is above `cushion + reservoir`, we use the highest available quality level.
 *
 * If the buffer is in the `cushion` interval, we use the first bitrate proportial with the
 * occupied buffer out of the cushion space, i.e. `(bufferLevel - reservoir) / cushion`.
 */
var BB = /** @class */ (function (_super) {
    __extends(BB, _super);
    function BB(video) {
        var _this = _super.call(this) || this;
        _this.bitrateArray = video.bitrateArray;
        _this.n = _this.bitrateArray.length;
        _this.bufferLevel = new BufferLevelGetter();
        return _this;
    }
    BB.prototype.getDecision = function (metrics, index, timestamp) {
        this.bufferLevel.update(metrics);
        var bufferLevel = this.bufferLevel.value;
        var bitrate = 0;
        var quality = 0;
        if (bufferLevel <= reservoir) {
            bitrate = this.bitrateArray[0];
        }
        else if (bufferLevel >= reservoir + cushion) {
            bitrate = this.bitrateArray[this.n - 1];
        }
        else {
            bitrate = this.bitrateArray[0] +
                (this.bitrateArray[this.n - 1] - this.bitrateArray[0]) *
                    (bufferLevel - reservoir) / cushion;
        }
        for (var i = this.n - 1; i >= 0; i--) {
            quality = i;
            if (bitrate >= this.bitrateArray[i]) {
                break;
            }
        }
        logger.log("bitrate ".concat(bitrate), "quality ".concat(quality), "buffer level ".concat(bufferLevel));
        return new Decision(index, quality, timestamp);
    };
    return BB;
}(AbrAlgorithm));
export { BB };
//# sourceMappingURL=bb.js.map