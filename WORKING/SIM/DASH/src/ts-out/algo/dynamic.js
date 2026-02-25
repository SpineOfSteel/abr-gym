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
 * Dynamic is the default DASH.js algorithm. This AbrAlgorithm class is, actually, just a
 * placeholder class.
 */
var Dynamic = /** @class */ (function (_super) {
    __extends(Dynamic, _super);
    function Dynamic() {
        return _super.call(this) || this;
    }
    Dynamic.prototype.getDecision = function (metrics, index, timestamp) {
        return new Decision(index, undefined, timestamp);
    };
    return Dynamic;
}(AbrAlgorithm));
export { Dynamic };
//# sourceMappingURL=dynamic.js.map