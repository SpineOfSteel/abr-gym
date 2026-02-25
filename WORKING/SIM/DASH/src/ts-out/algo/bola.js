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
import { Decision } from '../common/data';
/**
 * BOLA is a buffer-based algorithm provided by default in the DASH.js player. This AbrAlgorithm class
 * is, actually, just a placeholder class.
 */
var Bola = /** @class */ (function (_super) {
    __extends(Bola, _super);
    function Bola() {
        return _super.call(this) || this;
    }
    Bola.prototype.getDecision = function (metrics, index, timestamp) {
        return new Decision(index, undefined, timestamp);
    };
    return Bola;
}(AbrAlgorithm));
export { Bola };
//# sourceMappingURL=bola.js.map