//% weight=70 icon="\u30A2" color=#75CC05 block="NN"
namespace nn {

    //% blockId=nn_test
    //% block="Test"
    //% shim=nn::test
    export function test(): void {
    	basic.showString("sim-NN-test")
        return
    }

    /**
     * prints a string
     * @param text text to display, eg: "Hi, CPP!"
     */
     //% weight=92 blockGap=8
     //% block="Show|string %text" 
     //% blockId=nn_show
     //% shim=nn::show
     export function show(text: string): void {
        console.log("sim:" + text)
    	basic.showString("sim:" + text)
        return
	}

    //% blockId=nn_gettime
    //% block="Current Time" 
    //% shim=nn::gettime
	export function gettime(): string {
		return "sim:12:00:00"
	}

	//% blockId=nn_sumvec
	//% block="Sum Vec|number[] %vec"
	//% shim=nn::sumvec
	export function sumvec(vec: number[]): number {
        console.log("sim:" + vec[0])
        return vec[0]
	}
	
	//% blockId=nn_addvec
	//% block="Add Vec|number[] %vec1|number[] %vec2"
	//% shim=nn::addvec
	export function addvec(vec1: number[], vec2: number[]): number[] {
        console.log("sim:" + (vec1[0]+vec2[0]))
        return vec1
	}
	

}
